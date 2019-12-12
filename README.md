# proper-learning-SATD
Identify SATD using active learning and optimization technique


Software analytics can be improved by surveying; i.e. rechecking and (possibly) revising the labels offered by prior analysis. Surveying is a time-consuming task and effective surveyors must carefully manage their time. Specifically, they must balance the cost of further surveying against the additional benefits of that extra effort.
This paper proposes SURVEY0, an incremental Logistic Re- gression estimation method that implements cost/benefit analysis. Some classifier is used to rank the as-yet-unvisited examples according to how interesting they might be. Humans then review the most interesting examples, after which their feedback is used to update an estimator for estimating how many examples are remaining.
This paper evaluates SURVEY0 in the context of self-admitted technical debt. As software project mature, they can accumulate “technical debt” i.e. developer decisions which are sub-optimal and decrease the overall quality of the code. Such decisions are often commented on by programmers in the code; i.e. it is self- admitted technical debt (SATD). Recent results show that text classifiers can automatically detect such debt. We find that we can significantly outperform prior results by SURVEYing the data. Specifically, for ten open-source JAVA projects, we can find 83% of the technical debt via SURVEY0 using just 16% of the comments (and if higher levels of recall are required, SURVEY0 can adjust towards that with some additional effort).

Cite https://arxiv.org/pdf/1905.08297.pdf
