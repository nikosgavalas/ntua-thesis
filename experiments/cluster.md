| Algo | DatasetSize | Par | ExecTime(readSrc) | ExecTime(maps) |
| :--: | :---------: | :-: | :---------------: | :------------: |
|G|25|8|12|18|
|G|50|8|12|20|
|G|75|8|1|20|
|G|100|8|12,1,1,12|26,21,21,23|
|IF|25|8|11|22|
|IF|50|8|1|31|
|IF|75|8|12|42|
|IF|100|8|1|51|
|HS|25|8|12|31|
|HS|50|8|1|39|
|HS|75|8|11|58|
|HS|100|8|1|70|
|LD|25|8|1|22|
|LD|50|8|1|27|
|LD|75|8|12|44|
|LD|100|8|1|45|
|G|100|1|1|72|
|G|100|2|1|36|
|G|100|3|6|30|
|G|100|4|1|29|
|G|100|8|12,1,1,12|26,21,21,23|
|IF|100|1|1|110|***
|IF|100|2|12|96|
|IF|100|3|12|67|
|IF|100|4|1|61|
|IF|100|8|1|51|
|HS|100|1|6|156|
|HS|100|2|11|103,75,88,99|
|HS|100|3|1|60,160|
|HS|100|4|1|53,93|
|HS|100|8|12|69|
|LD|100|1|6|119|
|LD|100|2|1|71|
|LD|100|3|1|53|
|LD|100|4|12|78|
|LD|100|8|1|45|

** comma separated -> multiple runs of same setting
** execution is pipelined, meaning that total time = maps time.
** all times in seconds
** reads a file from one node (par 1), streams it through, time measured is time from start to end
** 25% -> 1,224,607 records
** 50% -> 2,449,215 records
** 75% -> 3,673,823 records
**100% -> 4,898,431 records
*** same node