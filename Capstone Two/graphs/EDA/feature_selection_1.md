**Basic Attempts / Successess / Rates**
*ex: throws, hucks, redzone*

 - Should only need to keep attempts and rate, as successes is just their product.
 - Correlation between attempts and successses is expected
 - drop with rate additions:
   - `<>_completions`
   - `<>_hucks_completed`
   - `<>_rz_scores`
 
**Stat leading to Stat**
*ex: `<>_d_blocks` leading to `<>_d_possessions`

 - will eventually remove the consequential stat, if it's basically a 1:1 relationship
   - in the above example it could be possesions, but the opposing team's throwaways also lead to posessions
   - a better example might be for `<home/away>_o_scores` to `<home/away>_d_points`
     - after scoring on offense, a team then starts on defense. excepting new time periods
 - prior to removal, those stats may be used in other derived features
   - so there may be a `<>_d_scores per <>_d_possessions`, and the dropped feature will still be captured in some way
 - list of stat to stats:
   - ...
   
**Points/Scores**

 - these are closely correlated as the points on offense/defense tend to be even for both teams, excepting a blowout
 - some are almost perfectly 1:1, as when "home" is on Offense, then "away" is on Defense
 - if everything is put into the persepctive of one team (home or away), then some of these can be dropped
   - ...
   
 
**Derived Rates - Inverse Relationships**

 - by definition, some of the added features are inverses of each other. should they be dropped?
   - `home/away_hold_rate` vs `away/home_break_rate`
     - **hold** when a team on offense scores
       - *everytime one team holds, the other team doesn't break*
     - **break** when a team on defense scores
       - *everytime one team breaks, the other team doesn't break*
     - investigation findings:
       - ***Additional Data Cleaning required!***. Some O/D score stats need to be adjusted.
   - `home/away_completion_rate` vs `away/home_block_rate`
     - should be slightly different due to throwaways, basically space between features is throwaways
	 - therefore this "space" should still be captured in model
   - `home/away_o_poss_per_hold` vs `away/home_d_poss_per_pt`
     - defined slightly differently when assessing potential derived features vs target features
	 - may decide to just keep one option during preprocessing
   
   
   
   
   
   
## Data Checks

 - throws > completions
 - home/away blocks > away/home turnovers