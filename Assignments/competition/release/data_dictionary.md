# Data dictionary

Japanese woodblock print auction sales. One row is one lot that sold.

`train.csv` has 12946 rows with prices, listed through 2024-02-13.

`test.csv` has the other 3222 lots with `final_bid` removed. You get every column
except the answer, so you can merge in anything you collected for those lots
just as you would for the training rows. Predict the price for each one.

About one lot in `test.csv` in sixteen is by an artist that appears nowhere in
`train.csv`, so make sure your code has something sensible to do with those. If your approach quietly falls
back to a global average for those, you should know that it does and say so.

| column | type | meaning |
|---|---|---|
| `row_id` | integer | identifier for this lot. Carries no information. |
| `link_text` | text | the lot title as the auction house wrote it. Often contains a subject, a place, and a year. |
| `artist_info` | text | the artist as the house lists them, usually with life dates in parentheses, for example `Hiroshige (1797 - 1858)`. |
| `series_info` | text | the print series, where the house named one. Missing for about half the lots. |
| `condition` | text | the house's free-text condition note. Mentions backing, creases, trimming, colour, and repairs. |
| `original` | 0 or 1 | 1 if the house calls it an original impression, 0 if a later reproduction. |
| `pulled_at` | timestamp | when the lot was captured during its listing. |
| `final_bid` | number | what it sold for, in US dollars. Take its log: your model returns `ln(final_bid)`. |
