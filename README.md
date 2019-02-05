# NarrativeFilter

This script detects the personal experiences in unsupervised data from social media based on supervised data. The second class performs topic modelling (NMF) on this data.

For determining the number of topics, the max of the TC-W2V can be used (this is the default) but a graph of the coherence values is also provided for manual evaluation.

Topic modelling based on the tutorial by Derek Greene: https://github.com/derekgreene/topic-model-tutorial/blob/master/3%20-%20Parameter%20Selection%20for%20NMF.ipynb

Topic coherence, measured using TC-W2V, was used to select the number of topics.

O’Callaghan,    D.,    Greene,    D.,    Carthy,    J.,    Cunningham,    P.:    Ananalysisofthecoherenceofdescriptorsintopicmodeling.ExpertSystemswithApplications42(13),5645–5657(2015).https://doi.org/10.1016/J.ESWA.2015.02.055
