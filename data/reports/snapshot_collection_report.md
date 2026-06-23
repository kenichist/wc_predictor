# Historical FIFA and World Football Elo Ranking Snapshots

## Overview

This project collects historical snapshots of FIFA men’s rankings and world football Elo ratings for use in World Cup prediction modelling.  Snapshots were compiled for dates close to major tournaments when credible full ranking tables were publicly available.  Each CSV file includes at least 30 teams and follows the format required by the data pipeline.

Due to the limited availability of archived tables and network restrictions, not all target dates yielded usable data.  The report below lists each snapshot collected, its source, the number of teams captured, and notes any missing dates or issues encountered.

## FIFA ranking snapshots collected

| Snapshot date | File name | Source and evidence | Teams collected | Notes |
|---|---|---|---|---|
| **17 May 2018** | `fifa_2018-05-17.csv` | Soccerphile’s “FIFA World Rankings May 2018” article enumerated the full ranking table released on **17 May 2018**, listing Germany (1), Brazil (2) and Belgium (3) and continuing down to San Marino at position 207【741682819120023†L92-L128】【741682819120023†L210-L299】. | 211 teams | Snapshot taken just before the **2018 World Cup**.  Points were not published in this listing, so the `points` column in the CSV is blank. |
| **4 Apr 2024** | `fifa_2024-04-04.csv` | Soccerphile’s “FIFA World Rankings April 2024” page provided the full ranking released on **4 April 2024**.  The article notes that World Cup winners Argentina remained top ahead of France, Belgium, England and Brazil【233245900350288†L69-L90】.  It lists all ranked teams from Argentina (1) to San Marino (210), plus unranked Eritrea【233245900350288†L98-L119】【233245900350288†L210-L309】. | 210 teams | Snapshot collected to represent the period before the **Euro/Copa America 2024** tournaments.  Points were not included in the published list; `points` is blank in the CSV. |
| **11 Jun 2026** | `fifa_2026-06-11.csv` | The 2026 ranking came from the “FIFA men’s world rankings” page on WhereIG, which republishes the official FIFA ranking released on **11 June 2026**.  The table shows Argentina first with 1877.27 points, followed by Spain and France, and runs down to San Marino at position 211【442015206949523†L145-L153】【442015206949523†L340-L356】. | 211 teams | Snapshot used for the **2026 World Cup**.  Points are included. |

**Missing FIFA snapshots:**

* **2010 (pre‑11 Jun)** – Soccerphile’s March 2010 page showed only the top 20 teams and referred readers to a full table that is no longer accessible.  No credible source with a full table (≥30 teams) for early 2010 could be found within the accessible data.
* **2014 (pre‑12 Jun)** – Soccerphile’s June 2014 and May 2014 pages only provide the top‑20 rankings, and the links to the full tables redirect to the generic rankings page.  Attempts to locate full tables on other sources (Football Ranking, Transfermarkt, etc.) were blocked by CAPTCHAs or unavailable.  As a result, a 2014 snapshot is not included.

## World Football Elo snapshots collected

| Snapshot date | File name | Source and evidence | Teams collected | Notes |
|---|---|---|---|---|
| **22 Jun 2026** | `elo_2026-06-22.csv` | International‑Football.net’s Elo ratings table (as of **22 June 2026**) replicates the rating calculations from Eloratings.net.  The page lists Spain in first place with 2 157 points, followed by Argentina, France, Portugal and Brazil【153271251787473†L355-L364】.  It provides an ordered list of teams with their Elo scores down to the lower ranked national teams【153271251787473†L355-L364】. | 211 teams | Used for the **2026 World Cup** snapshot. |

**Missing Elo snapshots:**

* Attempts were made to obtain Elo ratings for 2010, 2014, 2018, 2022 and 2024 using the date‑parameterized pages on International‑Football.net (e.g., `?year=2014&month=06&day=11`), but these pages repeatedly redirected to a placeholder file and did not load the data.  Without reliable access to the archived ratings, those snapshots could not be collected.

## Manual judgement and data issues

* **Points columns:**  Many Soccerphile ranking articles list the ranking order but not the points.  For those snapshots (2018 and 2024) the `points` column has been left blank rather than guessing values.
* **Team names normalization:**  The team names in the collected tables were used as published.  Future processing should normalize variants (e.g., “USA” to “United States”) as required.
* **Unranked teams:**  In the April 2024 FIFA list, Eritrea was noted as unranked【233245900350288†L210-L309】.  It was not included in the CSV as it lacks a rank.

## Conclusion

Despite the challenges of locating historical ranking tables and restrictions on dynamic websites, three FIFA ranking snapshots (2018‑05‑17, 2024‑04‑04 and 2026‑06‑11) and one world football Elo snapshot (2026‑06‑22) were successfully collected.  These snapshots cover recent tournaments and provide comprehensive team lists suitable for the prediction project.  Older snapshots (2010 and 2014) remain missing due to unavailability of credible full tables and should be revisited if new sources become accessible.
