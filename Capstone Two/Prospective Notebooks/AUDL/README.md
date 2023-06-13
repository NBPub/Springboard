# Brief Readme

## AUDL Data

US/Canada Professional men's ultimate frisbee league, active from 2012-*present*. Wealth of available statistics, some through public API.

 - **team game summary** - *starting with this*
 - roster + playtime
 - individual player stats
 - "play-by-play" with position tracking for frisbee
   - each attempted throw and related events
 - team information (location, colors, etc. . . )
 - more . . .
 
*Rules*

- games are played in 4 timed quarters
- starting possession changes ??? quarter/half
- incomplete throw is change of possession
- point scored by catching it in the endzone
- lots of terminology to be defined with data



***Tentative Goal(s)***

 - collect statistics for model in tables
   - query API / scrape pages --> save to DataFrame --> save to Parquet
   - persist final data files in repository for others to use
 - unravel "game-events" data return into play-by-play table, will have to decide on columns
 
 **Team Game Stats**
 
 - various summary stats for each team (Home, Away) and final scores
 - add various "derived" statistics, see preliminary EDA notebooks
 - ***PREDICT***
   - home margin: score difference between home and away team
   - home win: *boolean*, did the home team score more than the away team?
 - group by team name when evaluating distributions / dimensionality reduction(s)


### API

https://www.docs.audlstats.com/

#### Unlisted Endpoints

 - 

