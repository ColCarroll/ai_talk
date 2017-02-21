# Slides for 2-21-17 talk at the Boston AI Meetup

## Looking at the slides
1. Clone this repo

### ...with Python
2. Run `python -m http.server` (`python -m SimpleHTTPServer` on 2.x)
3. Open http://localhost:8000.

### ... with Node (4.0.0 or later) (also allows hot reloading)
2. `npm install`
3. `npm. start`
4. Open http://localhost:8000.

## Running the code

These slides started as an interactive Jupyter notebook, using the 
[https://github.com/damianavila/RISE](RISE project).  See website for 
installation.

There is a `requirements.txt` in the `talk_code` folder -- all 
code was written in python 3.6, but will probably work with 2.7, 3.4, 3.5.  

Also required is data from 
[https://www.kaggle.com/c/march-machine-learning-mania-2017](Kaggle) -- 
the scripts expect to see the two files 
  - RegularSeasonDetailedResults.csv
  - Teams.csv
in the same directory (`talk_code/`) as the scripts.  

Open `slides.ipynb` to run the code.  Some calculations are done lazily and 
cached, so it may be awful slow the first time through.
