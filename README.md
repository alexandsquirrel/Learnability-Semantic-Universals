# Learnability-Semantic-Universals
We attempt to reproduce the results from the paper Learnability and Semantic Universals (https://semanticsarchive.net/Archive/mQ2Y2Y2Z/LearnabilitySemanticUniversals.pdf). We also made improvements to the original data generation algorithm, as well as an attemp to probe language models to investigate what the model has learned.

# Setup
This project requires tensorflow 1.4 and python 2.7+ (note that recent python versions might not support this tensorflow version). It has been (non-extensively) tested that the code works for python 3.6 on Windows.\
Please run the following command to install the dependencies:
```
pip install -r dependencies.txt
```

# Running Experiencements
To run an experiment (for example, 1a), simply run the following command:
```
python quant_verify.py --exp one_a --out_path /tmp/exp1a/
```
where --out_path contains both the result files and the model checkpoints.

# Analyzing Data
To visualize the results, run the following command:
```
import analysis
analysis.experiment_one_a_analysis()
```

# Finally...
For more usage details, please refer to the authors' page:\
https://github.com/shanest/quantifier-rnn-learning \
Thank you for reading!
