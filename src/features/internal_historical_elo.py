from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import config_path, load_config, resolve_project_path
from src.io_utils import read_dataframe, stable_id, write_dataframe
from src.normalize import canonicalize_match_teams, classify_competition_type, normalize_team_name


logger = logging.getLogger(__name__)


INTERNAL_HISTORICAL_ELO_COLUMNS = [
    "home_internal_elo",
    "away_internal_elo",
    "internal_elo_diff",
    "home_internal_elo_rank",
    "away_internal_elo_rank",
    "internal_elo_rank_diff",
]

INTERNAL_HISTORICAL_ELO_HISTORY_COLUMNS = ["date", "team", "elo", "elo_rank", "matches_played", "source"]

SOURCE_NAME = "internal_historical_elo"


@dataclass(frozen=True)
class InternalEloSettings:
    default_elo: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 50.0
    goal_diff_cap: float = 4.0
    shootout_win_score: float = 0.75
    k_multipliers: dict[str, float] | None = None


@dataclass
class InternalEloBuildResult:
    features: pd.DataFrame
    history: pd.DataFrame
    settings: InternalEloSettings


def compute_internal_historical_elo(
    matches: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    shootouts: pd.DataFrame | None = None,
    default_elo: float | None = None,
    k_factor: float | None = None,
    home_advantage: float | None = None,
    goal_diff_cap: float | None = None,
    shootout_win_score: float | None = None,
    k_multipliers: dict[str, float] | None = None,
) -> InternalEloBuildResult:
    """Reconstruct leakage-safe pre-match Elo ratings from project match results."""
    cfg = config or load_config()
    settings = _settings_from_config(
        cfg,
        default_elo=default_elo,
        k_factor=k_factor,
        home_advantage=home_advantage,
        goal_diff_cap=goal_diff_cap,
        shootout_win_score=shootout_win_score,
        k_multipliers=k_multipliers,
    )
    if matches.empty:
        return InternalEloBuildResult(
            features=pd.DataFrame(columns=["match_id", "date", "home_team", "away_team", *INTERNAL_HISTORICAL_ELO_COLUMNS], index=matches.index),
            history=pd.DataFrame(columns=INTERNAL_HISTORICAL_ELO_HISTORY_COLUMNS),
            settings=settings,
        )

    df = matches.copy()
    df["row_id"] = np.arange(len(df))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "row_id"], kind="stable")
    shootout_lookup = _build_shootout_lookup(shootouts, cfg)

    ratings: dict[str, float] = {}
    matches_played: defaultdict[str, int] = defaultdict(int)
    feature_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []

    for match_date, day_matches in df.groupby("date", sort=True, dropna=False):
        teams_on_date = _teams_from_matches(day_matches)
        for team in teams_on_date:
            ratings.setdefault(team, settings.default_elo)

        pre_rankings = _rank_ratings(ratings)
        pending_deltas: defaultdict[str, float] = defaultdict(float)
        pending_matches: defaultdict[str, int] = defaultdict(int)

        for row in day_matches.itertuples(index=False):
            rec = row._asdict()
            home = _clean_team(rec.get("home_team"))
            away = _clean_team(rec.get("away_team"))
            home_pre = ratings.get(home, settings.default_elo) if home else np.nan
            away_pre = ratings.get(away, settings.default_elo) if away else np.nan
            home_rank = pre_rankings.get(home, np.nan) if home else np.nan
            away_rank = pre_rankings.get(away, np.nan) if away else np.nan

            feature_rows.append(
                {
                    "row_id": rec["row_id"],
                    "match_id": rec.get("match_id"),
                    "date": rec.get("date"),
                    "home_team": home,
                    "away_team": away,
                    "home_internal_elo": home_pre,
                    "away_internal_elo": away_pre,
                    "internal_elo_diff": home_pre - away_pre,
                    "home_internal_elo_rank": home_rank,
                    "away_internal_elo_rank": away_rank,
                    "internal_elo_rank_diff": home_rank - away_rank,
                }
            )

            home_score = rec.get("home_score")
            away_score = rec.get("away_score")
            if not home or not away or pd.isna(home_score) or pd.isna(away_score):
                continue

            neutral = _is_neutral(rec.get("neutral"))
            advantage = 0.0 if neutral else settings.home_advantage
            expected_home = _expected_home_score(float(home_pre), float(away_pre), advantage)
            shootout_key = _shootout_key(rec.get("date"), home, away)
            actual_home = _actual_home_score(
                float(home_score),
                float(away_score),
                shootout_winner=shootout_lookup.get(shootout_key) if shootout_key is not None else None,
                home_team=home,
                away_team=away,
                shootout_win_score=settings.shootout_win_score,
            )
            goal_multiplier = _goal_difference_multiplier(
                float(home_score),
                float(away_score),
                goal_diff_cap=settings.goal_diff_cap,
            )
            competition_type = rec.get("competition_type") or classify_competition_type(rec.get("tournament"))
            k = _match_k_factor(str(competition_type), settings) * goal_multiplier
            delta = k * (actual_home - expected_home)
            pending_deltas[home] += delta
            pending_deltas[away] -= delta
            pending_matches[home] += 1
            pending_matches[away] += 1

        for team, delta in pending_deltas.items():
            ratings[team] = ratings.get(team, settings.default_elo) + delta
        for team, count in pending_matches.items():
            matches_played[team] += count

        if pending_matches:
            post_rankings = _rank_ratings(ratings)
            for team in sorted(pending_matches):
                history_rows.append(
                    {
                        "date": match_date,
                        "team": team,
                        "elo": ratings[team],
                        "elo_rank": post_rankings[team],
                        "matches_played": matches_played[team],
                        "source": SOURCE_NAME,
                    }
                )

    features = pd.DataFrame(feature_rows).sort_values("row_id", kind="stable").drop(columns=["row_id"])
    features.index = matches.index
    history = pd.DataFrame(history_rows, columns=INTERNAL_HISTORICAL_ELO_HISTORY_COLUMNS)
    if not history.empty:
        history = history.sort_values(["date", "elo_rank", "team"], kind="stable").reset_index(drop=True)
    return InternalEloBuildResult(features=features, history=history, settings=settings)


def build_internal_historical_elo_artifacts(config: dict[str, Any] | None = None) -> InternalEloBuildResult:
    """Build internal Elo features, save rating history, update feature files, and write the report."""
    cfg = config or load_config()
    all_matches = load_internal_elo_match_base(cfg)
    shootouts = _load_optional_shootouts(cfg)
    result = compute_internal_historical_elo(all_matches, config=cfg, shootouts=shootouts)
    write_internal_historical_elo_history(result.history, cfg)
    _merge_internal_elo_into_feature_files(result.features, cfg)
    write_internal_elo_reconstruction_report(config=cfg, result=result)
    return result


def load_internal_elo_match_base(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load the broadest available match base for chronological rating reconstruction."""
    cfg = config or load_config()
    all_matches_path = _optional_config_path(cfg, "all_matches_clean")
    if all_matches_path is not None and all_matches_path.exists():
        try:
            return read_dataframe(all_matches_path)
        except Exception as exc:
            logger.warning("Could not read %s for internal Elo reconstruction: %s", all_matches_path, exc)

    frames: list[pd.DataFrame] = []
    for key in ("historical_matches_clean", "recent_matches_clean", "worldcup_2026_fixtures_clean"):
        path = _optional_config_path(cfg, key)
        if path is None or not path.exists():
            continue
        try:
            frame = read_dataframe(path)
        except Exception as exc:
            logger.warning("Could not read fallback match file %s: %s", path, exc)
            continue
        if not frame.empty:
            frames.append(frame)
    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        return _dedupe_match_base(combined.dropna(subset=["date", "home_team", "away_team"]))

    raw_frames = _load_raw_match_base(cfg)
    if raw_frames:
        combined = pd.concat(raw_frames, ignore_index=True, sort=False)
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        return _dedupe_match_base(combined.dropna(subset=["date", "home_team", "away_team"]))

    training_csv = _optional_config_path(cfg, "match_training_dataset_csv")
    if training_csv is not None and training_csv.exists():
        logger.warning("Falling back to %s; early pre-2010 history may be unavailable", training_csv)
        return read_dataframe(training_csv)
    raise FileNotFoundError("No match base is available for internal historical Elo reconstruction")


def write_internal_historical_elo_history(history: pd.DataFrame, config: dict[str, Any]) -> Path:
    return write_dataframe(history, _path_from_config(config, "internal_historical_elo", "data/features/internal_historical_elo.csv"))


def write_internal_elo_reconstruction_report(
    *,
    config: dict[str, Any] | None = None,
    result: InternalEloBuildResult | None = None,
) -> Path:
    cfg = config or load_config()
    history = result.history if result is not None else _read_optional_dataframe(_path_from_config(cfg, "internal_historical_elo", "data/features/internal_historical_elo.csv"))
    settings = result.settings if result is not None else _settings_from_config(cfg)
    training = _read_first_existing(
        [
            _optional_config_path(cfg, "match_training_dataset_advanced_parquet"),
            _optional_config_path(cfg, "match_training_dataset_parquet"),
        ]
    )
    prediction = _read_first_existing(
        [
            _optional_config_path(cfg, "worldcup_2026_prediction_input_advanced"),
            _optional_config_path(cfg, "worldcup_2026_prediction_input"),
        ]
    )
    ablation = _read_optional_dataframe(_path_from_config(cfg, "ablation_results_csv", "data/experiments/ablation_results.csv"))
    comparison = _ablation_comparison(ablation)

    lines = [
        "# Internal Historical Elo Reconstruction Report",
        "",
        "This report covers the `internal_historical_elo` feature group reconstructed from the project's historical international match results. It does not use external World Football Elo snapshots.",
        "",
        "## Configuration",
        "",
        f"- Starting Elo: `{settings.default_elo:.0f}`",
        f"- Base K-factor: `{settings.k_factor:.2f}`",
        f"- K-factors used: `{_format_k_factors(settings)}`",
        f"- Home advantage: `{settings.home_advantage:.2f}` Elo points for non-neutral matches",
        "- Neutral venue handling: home advantage is set to `0` when `neutral=True`.",
        f"- Goal-difference formula: `{_goal_difference_formula(settings.goal_diff_cap)}`",
        f"- Penalty shootout handling: drawn matches with a shootout winner use `{settings.shootout_win_score:.2f}` for the shootout winner and `{1.0 - settings.shootout_win_score:.2f}` for the loser.",
        "",
        "## Coverage",
        "",
        f"- Number of teams: `{_history_team_count(history)}`",
        f"- Date range: `{_history_date_range(history)}`",
        f"- Historical training coverage: `{_coverage(training):.3f}`",
        f"- World Cup 2026 coverage: `{_coverage(prediction):.3f}`",
        "",
        "## Validation Comparison",
        "",
        f"- Compared feature sets: `core_football_only` vs `core_plus_internal_elo`",
        f"- Improves log loss over `core_football_only`: `{comparison['improves']}`",
        f"- Log-loss delta (`core_plus_internal_elo` - `core_football_only`): `{comparison['delta']}`",
        f"- Notes: {comparison['notes']}",
        "",
    ]
    path = _path_from_config(cfg, "internal_elo_reconstruction_report_md", "data/reports/internal_elo_reconstruction_report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _settings_from_config(
    cfg: dict[str, Any],
    *,
    default_elo: float | None = None,
    k_factor: float | None = None,
    home_advantage: float | None = None,
    goal_diff_cap: float | None = None,
    shootout_win_score: float | None = None,
    k_multipliers: dict[str, float] | None = None,
) -> InternalEloSettings:
    base_cfg = cfg.get("elo", {})
    internal_cfg = cfg.get("internal_historical_elo", {})
    multipliers = dict(base_cfg.get("k_multipliers", {}))
    multipliers.update(internal_cfg.get("k_multipliers", {}))
    if k_multipliers:
        multipliers.update(k_multipliers)
    return InternalEloSettings(
        default_elo=float(default_elo if default_elo is not None else internal_cfg.get("default_elo", base_cfg.get("default_elo", 1500))),
        k_factor=float(k_factor if k_factor is not None else internal_cfg.get("k_factor", base_cfg.get("k_factor", 20))),
        home_advantage=float(home_advantage if home_advantage is not None else internal_cfg.get("home_advantage", base_cfg.get("home_advantage", 50))),
        goal_diff_cap=float(goal_diff_cap if goal_diff_cap is not None else internal_cfg.get("goal_diff_cap", 4)),
        shootout_win_score=float(shootout_win_score if shootout_win_score is not None else internal_cfg.get("shootout_win_score", 0.75)),
        k_multipliers=multipliers,
    )


def _match_k_factor(competition_type: str, settings: InternalEloSettings) -> float:
    multipliers = settings.k_multipliers or {}
    return settings.k_factor * float(multipliers.get(competition_type, multipliers.get("other", 1.0)))


def _expected_home_score(home_elo: float, away_elo: float, home_advantage: float) -> float:
    adjusted_home = home_elo + home_advantage
    return 1.0 / (1.0 + 10 ** ((away_elo - adjusted_home) / 400.0))


def _actual_home_score(
    home_score: float,
    away_score: float,
    *,
    shootout_winner: str | None,
    home_team: str,
    away_team: str,
    shootout_win_score: float,
) -> float:
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    if shootout_winner == home_team:
        return shootout_win_score
    if shootout_winner == away_team:
        return 1.0 - shootout_win_score
    return 0.5


def _goal_difference_multiplier(home_score: float, away_score: float, *, goal_diff_cap: float) -> float:
    goal_diff = min(abs(home_score - away_score), goal_diff_cap)
    if goal_diff <= 1:
        return 1.0
    return float(np.log(goal_diff + 1.0))


def _rank_ratings(ratings: dict[str, float]) -> dict[str, int]:
    if not ratings:
        return {}
    frame = pd.DataFrame({"team": list(ratings), "elo": list(ratings.values())})
    frame["elo_rank"] = frame["elo"].rank(method="min", ascending=False).astype(int)
    return dict(zip(frame["team"], frame["elo_rank"]))


def _teams_from_matches(matches: pd.DataFrame) -> set[str]:
    teams: set[str] = set()
    for column in ("home_team", "away_team"):
        if column not in matches.columns:
            continue
        teams.update(team for team in matches[column].map(_clean_team).dropna().tolist() if team)
    return teams


def _clean_team(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _is_neutral(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _load_optional_shootouts(cfg: dict[str, Any]) -> pd.DataFrame | None:
    path = _optional_config_path(cfg, "historical_shootouts_raw")
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning("Could not read historical shootout data from %s: %s", path, exc)
        return None


def _load_raw_match_base(cfg: dict[str, Any]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    historical_raw = _optional_config_path(cfg, "historical_results_raw")
    if historical_raw is not None and historical_raw.exists():
        try:
            frames.append(_load_raw_historical_results(historical_raw, cfg))
        except Exception as exc:
            logger.warning("Could not load raw historical results from %s: %s", historical_raw, exc)

    worldcup_raw = _optional_config_path(cfg, "worldcup_2026_raw")
    if worldcup_raw is not None and worldcup_raw.exists():
        try:
            from src.sources.worldcup_json import parse_worldcup_json

            frames.append(parse_worldcup_json(worldcup_raw))
        except Exception as exc:
            logger.warning("Could not load raw World Cup fixtures from %s: %s", worldcup_raw, exc)
    return [frame for frame in frames if not frame.empty]


def _load_raw_historical_results(path: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    output = pd.read_csv(path)
    output.columns = [str(column).strip().lower() for column in output.columns]
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(output.columns)
    if missing:
        raise ValueError(f"missing required raw historical columns: {sorted(missing)}")
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output["home_score"] = pd.to_numeric(output["home_score"], errors="coerce")
    output["away_score"] = pd.to_numeric(output["away_score"], errors="coerce")
    output = output.dropna(subset=["date", "home_team", "away_team"])
    output["neutral"] = output.get("neutral", pd.Series(False, index=output.index)).map(_is_neutral).fillna(False)
    mapping_path = _optional_config_path(cfg, "team_name_mapping")
    if mapping_path is not None and mapping_path.exists():
        output = canonicalize_match_teams(output, mapping_path=mapping_path)
    output["tournament"] = output.get("tournament", pd.Series("Other", index=output.index)).fillna("Other")
    output["competition_type"] = output["tournament"].map(classify_competition_type)
    output["source"] = "historical_results_raw"
    output["match_id"] = [
        stable_id(
            "historical",
            row.date.strftime("%Y-%m-%d"),
            row.home_team,
            row.away_team,
            row.home_score,
            row.away_score,
            row.tournament,
        )
        for row in output.itertuples(index=False)
    ]
    keep_columns = ["match_id", "source", "date", "home_team", "away_team", "neutral", "tournament", "competition_type", "home_score", "away_score"]
    return output[keep_columns].sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)


def _build_shootout_lookup(shootouts: pd.DataFrame | None, cfg: dict[str, Any]) -> dict[tuple[pd.Timestamp, str, str], str]:
    if shootouts is None or shootouts.empty:
        return {}
    required = {"date", "home_team", "away_team", "winner"}
    if not required.issubset(shootouts.columns):
        return {}
    prepared = shootouts.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    for column in ("home_team", "away_team", "winner"):
        prepared[column] = prepared[column].map(normalize_team_name)
    mapping_path = _optional_config_path(cfg, "team_name_mapping")
    if mapping_path is not None and mapping_path.exists():
        prepared = canonicalize_match_teams(prepared, mapping_path=mapping_path)
        prepared["winner"] = prepared["winner"].map(normalize_team_name)
    prepared = prepared.dropna(subset=["date", "home_team", "away_team", "winner"])
    return {
        _shootout_key(row.date, row.home_team, row.away_team): row.winner
        for row in prepared.itertuples(index=False)
    }


def _shootout_key(date: object, home_team: object, away_team: object) -> tuple[pd.Timestamp, str, str] | None:
    timestamp = pd.to_datetime(date, errors="coerce")
    if pd.isna(timestamp):
        return None
    return (pd.Timestamp(timestamp).normalize(), str(home_team), str(away_team))


def _merge_internal_elo_into_feature_files(features: pd.DataFrame, cfg: dict[str, Any]) -> None:
    training_path = _first_existing_path(
        [
            _optional_config_path(cfg, "match_training_dataset_advanced_parquet"),
            _optional_config_path(cfg, "match_training_dataset_parquet"),
        ]
    )
    prediction_path = _first_existing_path(
        [
            _optional_config_path(cfg, "worldcup_2026_prediction_input_advanced"),
            _optional_config_path(cfg, "worldcup_2026_prediction_input"),
        ]
    )
    if training_path is None or prediction_path is None:
        logger.info("Skipping internal Elo feature-file merge because baseline feature files are missing")
        return

    training = _merge_features(read_dataframe(training_path), features)
    try:
        prediction = _merge_features(read_dataframe(prediction_path), features)
    except Exception as exc:
        logger.warning("Skipping internal Elo merge into prediction input because %s could not be read: %s", prediction_path, exc)
        prediction = pd.DataFrame()
    write_dataframe(training, _path_from_config(cfg, "match_training_dataset_advanced_parquet", "data/features/match_training_dataset_advanced.parquet"))
    write_dataframe(training, _path_from_config(cfg, "match_training_dataset_advanced_csv", "data/features/match_training_dataset_advanced.csv"))
    if not prediction.empty:
        prediction_path_out = _path_from_config(cfg, "worldcup_2026_prediction_input_advanced", "data/features/worldcup_2026_prediction_input_advanced.parquet")
        write_dataframe(prediction, prediction_path_out)
        write_dataframe(prediction, prediction_path_out.with_suffix(".csv"))


def _merge_features(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    output = df.drop(columns=[column for column in INTERNAL_HISTORICAL_ELO_COLUMNS if column in df.columns], errors="ignore")
    if "match_id" in output.columns:
        output = output.drop_duplicates("match_id", keep="first")
    feature_frame = features[["match_id", *INTERNAL_HISTORICAL_ELO_COLUMNS]].drop_duplicates("match_id", keep="last")
    merged = output.merge(feature_frame, on="match_id", how="left")
    missing = merged[INTERNAL_HISTORICAL_ELO_COLUMNS].isna().any(axis=1)
    if not missing.any() or not {"date", "home_team", "away_team"}.issubset(features.columns):
        return merged

    fallback_features = features[["date", "home_team", "away_team", *INTERNAL_HISTORICAL_ELO_COLUMNS]].copy()
    fallback_features["_internal_elo_key"] = _match_key(fallback_features)
    fallback_features = fallback_features.dropna(subset=["_internal_elo_key"]).drop_duplicates("_internal_elo_key", keep="last")
    fallback_lookup = fallback_features.set_index("_internal_elo_key")[INTERNAL_HISTORICAL_ELO_COLUMNS]
    fallback_values = fallback_lookup.reindex(_match_key(merged.loc[missing]))
    fallback_values.index = merged.index[missing]
    for column in INTERNAL_HISTORICAL_ELO_COLUMNS:
        merged.loc[missing, column] = merged.loc[missing, column].combine_first(fallback_values[column])
    still_missing = merged[INTERNAL_HISTORICAL_ELO_COLUMNS].isna().any(axis=1)
    if still_missing.any():
        merged = _fill_missing_from_team_history(merged, features, still_missing)
    return merged


def _match_key(df: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    home = df["home_team"].astype("string").str.strip()
    away = df["away_team"].astype("string").str.strip()
    key = dates + "|" + home + "|" + away
    return key.where(dates.notna() & home.notna() & away.notna())


def _fill_missing_from_team_history(df: pd.DataFrame, features: pd.DataFrame, missing: pd.Series) -> pd.DataFrame:
    if not {"date", "home_team", "away_team"}.issubset(df.columns):
        return df
    output = df.copy()
    lookup = _team_feature_lookup(features)
    if not lookup:
        return output

    for idx, row in output.loc[missing].iterrows():
        match_date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(match_date):
            continue
        for side in ("home", "away"):
            team = _clean_team(row.get(f"{side}_team"))
            values = _latest_team_feature(lookup, team, pd.Timestamp(match_date))
            if values is None:
                continue
            output.at[idx, f"{side}_internal_elo"] = output.at[idx, f"{side}_internal_elo"] if pd.notna(output.at[idx, f"{side}_internal_elo"]) else values["elo"]
            output.at[idx, f"{side}_internal_elo_rank"] = (
                output.at[idx, f"{side}_internal_elo_rank"] if pd.notna(output.at[idx, f"{side}_internal_elo_rank"]) else values["rank"]
            )
        if pd.notna(output.at[idx, "home_internal_elo"]) and pd.notna(output.at[idx, "away_internal_elo"]):
            output.at[idx, "internal_elo_diff"] = output.at[idx, "home_internal_elo"] - output.at[idx, "away_internal_elo"]
        if pd.notna(output.at[idx, "home_internal_elo_rank"]) and pd.notna(output.at[idx, "away_internal_elo_rank"]):
            output.at[idx, "internal_elo_rank_diff"] = output.at[idx, "home_internal_elo_rank"] - output.at[idx, "away_internal_elo_rank"]
    return output


def _team_feature_lookup(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if not {"date", "home_team", "away_team"}.issubset(features.columns):
        return {}
    frames = []
    for side in ("home", "away"):
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(features["date"], errors="coerce"),
                    "team": features[f"{side}_team"].map(_clean_team),
                    "elo": pd.to_numeric(features[f"{side}_internal_elo"], errors="coerce"),
                    "rank": pd.to_numeric(features[f"{side}_internal_elo_rank"], errors="coerce"),
                }
            )
        )
    long = pd.concat(frames, ignore_index=True).dropna(subset=["date", "team", "elo", "rank"])
    long = long.sort_values(["team", "date"], kind="stable").drop_duplicates(["team", "date"], keep="last")
    return {team: group.reset_index(drop=True) for team, group in long.groupby("team")}


def _latest_team_feature(lookup: dict[str, pd.DataFrame], team: str | None, match_date: pd.Timestamp) -> dict[str, float] | None:
    if not team:
        return None
    history = lookup.get(team)
    if history is None or history.empty:
        return None
    idx = history["date"].searchsorted(match_date, side="left") - 1
    if idx < 0:
        return {"elo": 1500.0, "rank": float(history["rank"].max())}
    row = history.iloc[idx]
    return {"elo": float(row["elo"]), "rank": float(row["rank"])}


def _dedupe_match_base(matches: pd.DataFrame) -> pd.DataFrame:
    return matches.sort_values(["date", "match_id"], kind="stable").drop_duplicates("match_id", keep="first").reset_index(drop=True)


def _coverage(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    required = [column for column in INTERNAL_HISTORICAL_ELO_COLUMNS if column in df.columns]
    if len(required) != len(INTERNAL_HISTORICAL_ELO_COLUMNS):
        return 0.0
    return float(df[required].notna().all(axis=1).mean())


def _ablation_comparison(ablation: pd.DataFrame) -> dict[str, str]:
    if ablation.empty or "feature_set" not in ablation.columns or "log_loss" not in ablation.columns:
        return {"improves": "unknown", "delta": "not available", "notes": "Run `python -m src.cli run-ablation` after rebuilding internal Elo features."}
    core = ablation[ablation["feature_set"].eq("core_football_only")]
    internal = ablation[ablation["feature_set"].eq("core_plus_internal_elo")]
    if core.empty or internal.empty:
        return {"improves": "unknown", "delta": "not available", "notes": "`core_plus_internal_elo` has not yet been compared against `core_football_only` in ablation results."}
    delta = float(internal.iloc[0]["log_loss"] - core.iloc[0]["log_loss"])
    improves = delta < 0
    return {
        "improves": str(improves),
        "delta": f"{delta:.6f}",
        "notes": "Lower log loss is better.",
    }


def _format_k_factors(settings: InternalEloSettings) -> str:
    multipliers = settings.k_multipliers or {"other": 1.0}
    parts = [f"{name}={settings.k_factor * float(multiplier):.2f}" for name, multiplier in sorted(multipliers.items())]
    return ", ".join(parts)


def _goal_difference_formula(goal_diff_cap: float) -> str:
    return f"1.0 for one-goal margins; ln(min(abs(goal_diff), {goal_diff_cap:.0f}) + 1.0) for larger margins"


def _history_team_count(history: pd.DataFrame) -> int:
    if history.empty or "team" not in history.columns:
        return 0
    return int(history["team"].nunique(dropna=True))


def _history_date_range(history: pd.DataFrame) -> str:
    if history.empty or "date" not in history.columns:
        return "not available"
    dates = pd.to_datetime(history["date"], errors="coerce").dropna()
    if dates.empty:
        return "not available"
    return f"{dates.min().date()} to {dates.max().date()}"


def _read_optional_dataframe(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return read_dataframe(path)


def _read_first_existing(paths: list[Path | None]) -> pd.DataFrame:
    path = _first_existing_path(paths)
    if path is None:
        return pd.DataFrame()
    return read_dataframe(path)


def _first_existing_path(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def _optional_config_path(cfg: dict[str, Any], key: str) -> Path | None:
    try:
        return config_path(cfg, key)
    except KeyError:
        return None


def _path_from_config(cfg: dict[str, Any], key: str, fallback: str) -> Path:
    configured = _optional_config_path(cfg, key)
    if configured is not None:
        return configured
    if key == "internal_elo_reconstruction_report_md":
        recommendation = _optional_config_path(cfg, "final_model_recommendation_md")
        if recommendation is not None:
            return recommendation.parent / "internal_elo_reconstruction_report.md"
    return resolve_project_path(fallback)
