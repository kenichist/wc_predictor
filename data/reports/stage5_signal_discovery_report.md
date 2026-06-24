# Stage 5 Signal Discovery Report

## Executive Summary

- Stage 5 status: `Stage 5 not achieved`.
- This is research/evaluation output only.
- Official production model logic was not changed.
- Negative deltas mean the official model is better than bookmaker odds-only.
- Paired official/bookmaker matches analyzed: `188`.
- Statistically meaningful winning segments found: `0`.

## Overall Official vs Bookmaker

| n_matches | official_log_loss | bookmaker_log_loss | mean_log_loss_delta | official_brier | bookmaker_brier | brier_delta | official_rps | bookmaker_rps | rps_delta | official_accuracy | bookmaker_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 188 | 1.005457 | 0.966467 | 0.038990 | 0.595706 | 0.569686 | 0.026020 | 0.212676 | 0.201729 | 0.010947 | 0.510638 | 0.553191 |

## Top Winning Segments

| segment_name | segment_value | n_matches | official_log_loss | bookmaker_log_loss | mean_log_loss_delta | ci_lower | ci_upper | statistically_meaningful | official_accuracy | bookmaker_accuracy | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| favorite_won_status | favorite_failed | 84 | 1.422863 | 1.453870 | -0.031007 | -0.104199 | 0.043849 | False | 0.119048 | 0.000000 | official_better_mean_ci_crosses_zero |
| official_confidence_bucket | 0.50-0.60 | 50 | 1.002918 | 1.021337 | -0.018419 | -0.098913 | 0.056569 | False | 0.540000 | 0.520000 | official_better_mean_ci_crosses_zero |
| bookmaker_confidence_bucket | 0.60-0.70 | 40 | 0.941479 | 0.950196 | -0.008717 | -0.083664 | 0.062061 | False | 0.575000 | 0.600000 | official_better_mean_ci_crosses_zero |
| close_match_probability_bucket | clear_0.60-0.70 | 40 | 0.941479 | 0.950196 | -0.008717 | -0.083664 | 0.062061 | False | 0.575000 | 0.600000 | official_better_mean_ci_crosses_zero |

## Top Losing Segments

| segment_name | segment_value | n_matches | official_log_loss | bookmaker_log_loss | mean_log_loss_delta | ci_lower | ci_upper | statistically_meaningful | official_accuracy | bookmaker_accuracy | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_confidence_bucket | 0.60-0.70 | 37 | 0.909661 | 0.791301 | 0.118360 | 0.038312 | 0.213210 | False | 0.648649 | 0.648649 | strong_bookmaker_advantage |
| favorite_won_status | favorite_won | 104 | 0.668320 | 0.572796 | 0.095525 | 0.059427 | 0.135114 | False | 0.826923 | 1.000000 | strong_bookmaker_advantage |
| bookmaker_confidence_bucket | 0.00-0.40 | 32 | 1.201446 | 1.116358 | 0.085088 | -0.035246 | 0.204886 | False | 0.343750 | 0.281250 | bookmaker_better_mean_ci_crosses_zero |
| close_match_probability_bucket | very_close_<=0.40 | 32 | 1.201446 | 1.116358 | 0.085088 | -0.035246 | 0.204886 | False | 0.343750 | 0.281250 | bookmaker_better_mean_ci_crosses_zero |
| tournament_year | 2022 | 63 | 1.087986 | 1.010652 | 0.077334 | -0.001766 | 0.155568 | False | 0.476190 | 0.523810 | bookmaker_better_mean_ci_crosses_zero |
| prediction_agreement | different_prediction | 34 | 1.093951 | 1.025362 | 0.068589 | -0.058846 | 0.185851 | False | 0.294118 | 0.529412 | bookmaker_better_mean_ci_crosses_zero |
| stage | round_of_16 | 23 | 0.853336 | 0.787686 | 0.065650 | -0.000891 | 0.131149 | False | 0.608696 | 0.652174 | bookmaker_better_mean_ci_crosses_zero |
| actual_result_type | draw | 40 | 1.399248 | 1.337922 | 0.061326 | -0.012959 | 0.132605 | False | 0.050000 | 0.000000 | bookmaker_better_mean_ci_crosses_zero |
| bookmaker_confidence_bucket | 0.70-1.00 | 27 | 0.813993 | 0.754462 | 0.059531 | -0.046519 | 0.161537 | False | 0.740741 | 0.740741 | bookmaker_better_mean_ci_crosses_zero |
| close_match_probability_bucket | heavy_>0.70 | 27 | 0.813993 | 0.754462 | 0.059531 | -0.046519 | 0.161537 | False | 0.740741 | 0.740741 | bookmaker_better_mean_ci_crosses_zero |

## Worst Official-Model Misses

| date | tournament_year | stage | home_team | away_team | actual_result | official_log_loss | bookmaker_log_loss | log_loss_delta | possible_failure_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-11-28 | 2022 | group_stage | South Korea | Ghana | away_win | 2.207921 | 1.079133 | 1.128788 | official assigned much lower probability to actual outcome |
| 2018-06-17 | 2018 | group_stage | Costa Rica | Serbia | away_win | 1.322009 | 0.636049 | 0.685960 | bookmaker selected actual outcome while official selected draw |
| 2022-11-25 | 2022 | group_stage | Qatar | Senegal | away_win | 1.255806 | 0.582982 | 0.672823 | bookmaker selected actual outcome while official selected home_win |
| 2022-12-02 | 2022 | group_stage | Cameroon | Brazil | home_win | 2.868255 | 2.246416 | 0.621839 | official overconfident wrong-side prediction |
| 2022-11-30 | 2022 | group_stage | Tunisia | France | home_win | 2.366697 | 1.818099 | 0.548598 | official overconfident wrong-side prediction |
| 2018-07-10 | 2018 | semifinal | France | Belgium | home_win | 1.495201 | 0.987715 | 0.507486 | bookmaker selected actual outcome while official selected away_win |
| 2014-07-01 | 2014 | round_of_16 | Belgium | United States | home_win | 1.230311 | 0.736456 | 0.493855 | bookmaker selected actual outcome while official selected away_win |
| 2022-12-18 | 2022 | final | Argentina | France | draw | 1.679004 | 1.189337 | 0.489667 | official underweighted draw |
| 2018-06-26 | 2018 | group_stage | Denmark | France | draw | 1.535925 | 1.052238 | 0.483687 | official underweighted draw |
| 2018-06-15 | 2018 | group_stage | Portugal | Spain | draw | 1.656697 | 1.213146 | 0.443551 | official underweighted draw |
| 2018-06-16 | 2018 | group_stage | Peru | Denmark | away_win | 1.305497 | 0.865120 | 0.440377 | bookmaker selected actual outcome while official selected home_win |
| 2022-11-29 | 2022 | group_stage | Ecuador | Senegal | away_win | 1.650003 | 1.211869 | 0.438134 | official overconfident wrong-side prediction |
| 2018-06-14 | 2018 | group_stage | Russia | Saudi Arabia | home_win | 0.838607 | 0.406558 | 0.432049 | bookmaker selected actual outcome while official selected draw |
| 2022-12-01 | 2022 | group_stage | Costa Rica | Germany | away_win | 0.563598 | 0.137588 | 0.426011 | official assigned much lower probability to actual outcome |
| 2018-06-21 | 2018 | group_stage | Denmark | Australia | draw | 1.662753 | 1.258890 | 0.403863 | official underweighted draw |
| 2022-11-27 | 2022 | group_stage | Croatia | Canada | home_win | 1.234596 | 0.839462 | 0.395134 | bookmaker selected actual outcome while official selected away_win |
| 2014-06-25 | 2014 | group_stage | Bosnia and Herzegovina | Iran | home_win | 1.205085 | 0.821528 | 0.383557 | bookmaker selected actual outcome while official selected away_win |
| 2022-11-27 | 2022 | group_stage | Belgium | Morocco | away_win | 1.912721 | 1.539479 | 0.373242 | official overconfident wrong-side prediction |
| 2022-12-02 | 2022 | group_stage | South Korea | Portugal | home_win | 1.844461 | 1.498411 | 0.346049 | official overconfident wrong-side prediction |
| 2022-11-27 | 2022 | group_stage | Spain | Germany | draw | 1.610628 | 1.270735 | 0.339893 | official underweighted draw |

## Feature Hypotheses

| failure_pattern | evidence_segment | proposed_feature_or_model_change | expected_help | leakage_risk | data_needed | priority |
| --- | --- | --- | --- | --- | --- | --- |
| draw handling | actual_result_type=draw (n=40, delta=0.061326) | draw-specific calibration or draw-aware ordinal/scoreline layer | Reduce overconfident non-draw probabilities when historical tournament draws are underweighted. | Low if calibrated only on prior tournament folds. | Historical no-leakage match probabilities and draw outcomes by tournament year. | high |
| favorite or underdog upset handling | worst_match=Nigeria vs Iceland (favorite failed or upset pattern, delta=0.306771) | underdog/favorite upset feature using disagreement between model, market, Elo, and form | Improve cases where the model is too confident in the wrong side of an upset. | Medium; all market/rating inputs must be pre-match as-of data. | Pre-match market odds, internal Elo, ranking snapshots, and opponent-adjusted form. | high |
| market low-confidence segment modeling | close_match_probability_bucket=very_close_<=0.40 (n=32, delta=0.085088) | separate calibration for low market-confidence matches | Target matches where bookmaker probabilities are flat and model disagreement may contain signal. | Low if bucket thresholds are fixed or tuned on prior folds only. | Complete historical 1X2 odds coverage and prior-fold calibration rows. | medium |
| opponent-adjusted recent form | official_confidence_bucket=0.60-0.70 (n=37, delta=0.118360) | strength-of-schedule adjusted rolling form features | Separate strong recent form against weak opponents from stronger signal against elite opponents. | Low if rolling windows close before match date. | Historical results with internal Elo/opponent strength as-of match date. | high |
| rest/travel/venue context missing | official_confidence_bucket=0.60-0.70 (n=37, delta=0.118360) | rest days, travel distance, host/venue effect, and tournament-stage interaction features | Capture World Cup-specific fatigue and venue context not present in generic match form. | Medium; fixture schedule and venue data must be known before match kickoff. | Venue locations, team travel bases, match kickoff dates, rest days, host flags. | medium |
| confederation matchup effects | official_confidence_bucket=0.60-0.70 (n=37, delta=0.118360) | confederation matchup and inter-confederation tournament calibration | Model cross-region matchup priors that generic team ratings may miss. | Low if team confederations are static metadata. | Team-to-confederation mapping for all historical World Cup teams. | medium |
| Elo/FIFA disagreement | official_confidence_bucket=0.60-0.70 (n=37, delta=0.118360) | feature capturing disagreement between internal Elo, FIFA/ranking, and market favorite | Identify matches where independent rating systems disagree and raw football model may be overconfident. | Medium; ranking snapshots must be as-of pre-match. | Internal Elo and FIFA/ranking snapshots joined to historical tournament matches. | medium |
| low-scoring match structure | official_confidence_bucket=0.60-0.70 (n=37, delta=0.118360) | defensive-strength and proper Dixon-Coles scoreline model | Improve 1X2 probabilities for low-scoring matches where draw and one-goal outcomes dominate. | Low if trained only on pre-tournament history. | Historical goals, team attack/defense strengths, and time-decayed scoreline fitting. | high |

## Optional Segment Availability

- Elo difference buckets: `False`.
- FIFA/ranking difference buckets: `False`.
- Confederation matchup buckets: `False`.
- Team buckets: `True`.

## Input File Check

- Stage 5 benchmark metrics present: `True`.
- Stage 5 calibrated market predictions present: `True`.
- Stage 5 calibrated market metrics present: `True`.
- Stage 5 benchmark report present: `True`.
- Stage 5 calibrated market report present: `True`.

## Recommendation

- No segment shows statistically meaningful official-model value over bookmaker odds under the current CI rule.
- Biggest failure segment: `official_confidence_bucket=0.60-0.70` with mean log-loss delta `0.118360`.
- Best next hypothesis to test: `draw-specific calibration or draw-aware ordinal/scoreline layer`.
- Keep production unchanged and test feature/model changes only through no-leakage historical folds.
