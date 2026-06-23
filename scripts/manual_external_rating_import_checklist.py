from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIFA_TEMPLATE = PROJECT_ROOT / "data" / "external" / "fifa_rankings_user_input_template.csv"
ELO_TEMPLATE = PROJECT_ROOT / "data" / "external" / "world_football_elo_user_input_template.csv"


def ensure_templates() -> None:
    FIFA_TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
    if not FIFA_TEMPLATE.exists():
        FIFA_TEMPLATE.write_text("date,team,rank,points,source,retrieved_at\n", encoding="utf-8")
    if not ELO_TEMPLATE.exists():
        ELO_TEMPLATE.write_text("date,team,elo,source,retrieved_at\n", encoding="utf-8")


def main() -> int:
    ensure_templates()
    print("Manual external rating import checklist")
    print()
    print("1. Download or copy the current FIFA men's ranking table from:")
    print("   https://inside.fifa.com/fifa-world-ranking/men")
    print("   Required output columns: date, team, rank, points, source, retrieved_at")
    print("   Accepted input headers include: country/team/nation, rank/position, points/total_points/rating, date/ranking_date/release_date")
    print(f"   User-fillable template: {FIFA_TEMPLATE}")
    print("   Import command:")
    print("   python -m src.cli acquire-fifa-rankings --input \"C:\\Users\\kenic\\Downloads\\fifa_rankings.csv\"")
    print("   Replace the example path with the actual file path you downloaded.")
    print()
    print("2. Download or copy the current World Football Elo table from:")
    print("   https://www.eloratings.net/")
    print("   Required output columns: date, team, elo, source, retrieved_at")
    print("   Accepted input headers include: country/team/nation, elo/rating/points, date/rating_date")
    print(f"   User-fillable template: {ELO_TEMPLATE}")
    print("   Import command:")
    print("   python -m src.cli acquire-world-football-elo --input \"C:\\Users\\kenic\\Downloads\\world_football_elo.csv\"")
    print("   Replace the example path with the actual file path you downloaded.")
    print()
    print("3. Validate:")
    print("   python -m src.cli validate-external-data")
    print()
    print("4. Rebuild features:")
    print("   python -m src.cli build-advanced-features")
    print()
    print("5. Check activation:")
    print("   python -m src.cli feature-null-report")
    print()
    print("6. Rerun ablation:")
    print("   python -m src.cli run-ablation")
    print()
    print("These templates are user input templates only. They are not real rating data and must not be used as model data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
