# Project2














CS-634 Data Mining 
Spring 2021
Project 2: Why should I trust you? 
Group Members: 
Yorleydis Oliveros, Huong Ly Ngo, Rob Mullaney
LIME Tutorial 
























Introduction:

	With Machine Learning at the center of recent advances in Science and Technology, a contentious relationship between any given Machine Learning model and its respective user will impede the wide-scale adoption (and benefits) of Machine Learning technology. Therefore, whether humans are using Machine Learning classifiers as tools or are deploying Machine Learning models within other products, the LIME methodology is essential to not only build trust between end-user and model, but to provide transparency by providing a qualitative understanding of the relationship between an instance/case and the model’s prediction. Ultimately, we argue that by providing explanations of features for individual predictions through the LIME and SP-LIME methods, then measuring the impact of these explanations on user trust and associated tasks, should assist in the adoption of Machine Learning models. 
	First off, when we provide “explanations for individual predictions”, it is important to know the background of the end-user. The reason being that Humans usually have some prior knowledge about the domain of any given Machine Learning model. Therefore, we want to empower an individual’s prior domain knowledge to either accept or reject a model’s prediction. However, it is important to provide clear and coherent explanations that are interpretable by the end-user, given the user’s limitations. Explaining predictions in an interpretable manner is an essential step in building the trust to properly use a Machine Learning model. 
	When providing an explanation for a prediction, there are a few essential criteria that must be met. First, the explanation of features must be interpretable. As mentioned before, this trait must take into account user limitations, but must provide some type of understanding between the input variables and prediction. Secondly, our explanation must be Locally Faithful. Although it is nearly impossible to provide an explanation that’s faithful without providing a complete description of the model itself, the explanation must highlight the primary features of the model within the vicinity of the prediction. Additionally, an explainer should be model-agnostic and be applicable to any model. Finally, our explanation should provide a Global Perspective in order to continue building trust with the user. Accuracy measurements are not always ideal when evaluating a model. Therefore, we want to explain the model while utilizing a select few explanations from individual predictions that are representative of the model. 
	The LIME method seeks to explain a classifier’s predictions by learning an interpretable model locally around the prediction. The explanation produced by LIME is obtained through the below formula. 

	It’s important to note that the formula can be used with different explanation families (G), fidelity functions (L), and complexity measures (Ω). Likewise, the constant g ( g ∈ G ) is a member of the set G, with G being a potentially interpretable model or explanation for families, such as linear models, decision trees, or a falling rules list. However, the LIME method only provides local explanations of our model, and if we were searching for a global explanation of the model, then the SP-LIME method would need to be utilized, which will be covered following the explanation and tutorial of the LIME method.
	When using LIME, we want to minimize the locality-aware loss (L) while maintaining model-agnosticity, and therefore, we cannot make any assumptions about f. We interpret the locality-aware loss function (L) as a measure of how unfaithful the constant g is when approximating f within the locality as defined by variable π. Variable π(z) is used as a proximity measure within an instance of z to x, in an attempt to define a prediction’s locality around x. Thus, as interpretable inputs vary, we estimate the locality-aware loss (L) by drawing samples weighted by variable π(z). All the while, keeping the complexity measures (Ω) low enough to be interpretable by humans. 
	As we minimize the locality-aware loss (L), we sample instances around x’ by drawing nonzero elements of x’ at random. Given sample  z  ∈ {0, 1}, which contains a share of nonzero elements of x’, we obtain the model’s original sample (z ∈ R) and are able to derive f(z), which is then used as a label for the explanation. 
	As previously mentioned, LIME provides an explanation for a single prediction, but if we were to search for an explanation of the model as a whole, then we would need to utilize the SP-LIME method instead. In this approach, a global explanation of the model is provided through a set of individual instances. 
	When using the SP-LIME method to explain a model as a whole, the instances used must be carefully selected since users are unable to examine an infinitely large number of explanations. Therefore, given a set of X instances, we start with a “Pick Step” of selecting B instances for the user to select, with B representing the budgeted number of explanations a user is willing to analyze. The “Pick Step” takes into account the explanations associated with each prediction, and should cover a diverse set of explanations that are representative of how the model behaves globally. 
	Given the explanations for a set of X instances (|X|=n), we construct an n * d’ explanation matrix (W), representing the number of the local importance of the interpretable features for each instance. Similarly, for each column (j) in W, we represent the global importance of the given column (j) by an importance score (I) within the explanation matrix. When selecting an importance score (I), we want to see features that explain many different instances to hold higher importance scores (I), but still avoid selecting instances with similar explanations and features.
	To avoid redundant coverage of features, we define a coverage function (c) which calculates the total importance of features appearing in at least one instance within a new set V, given the explanation matrix (W) and importance score (I). The coverage function is defined below, along with the algebraic goal of the “Pick Step” which is to find the set of W and I, which maximizes the coverage function. 
	Leveraging submodularity, an algorithm that iteratively adds the instance with the highest marginal coverage gain to the solution, offers a constant-factor approximation guarantee of 1-1/e to the optimum, optimizing our algorithm. 


	Finally, we conduct simulated user experiments in an attempt to evaluate the effect of explanations on trust between the end-user and model. Specifically, we seek to answer whether the explanations are faithful to the model, whether the explanations aid users in cementing trust of the model’s predictions, and whether the explanations are helpful when evaluating the model as a whole. 
	Within “Why Should I Trust You?” Explaining the Predictions of Any Classifier by Ribiero, et. al., sentiment analysis was performed on two separate datasets (books and DVDs) to classify product reviews as either positive or negative by using Decision Trees, Logistic Regression, Nearest Neighbors, or a Random Forest, while dividing each dataset with 75% of instances going into a train set and the remaining 25% of instances going into a test set. To explain individual predictions, the LIME approach is compared with the parzen method, and explains individual predictions by taking the gradient of the prediction probability function.  -------------------------- 






focuses on identifying an interpretable model over the interpretable representation that is locally faithful to the classifier. The below formula
  
	
	
	
____________________________________________________________________


The LIME (Local Interpretable model-agnostic explanation) technique - explains any classifiers' predictions by learning an interpretable model locally around the prediction.  
 
A benefit of using LIME is identifying an interpretable model over an interpretable representation that is locally faithful to the classifier.[1]  An interpretable model is defined as a qualitative understanding between the input variables and the response. When an interpretable model is locally faithful, the criterion must correspond to how the model behaves in the area in which it’s being predicted.
The following formula can be used to obtain LIME (Local Interpretable model-agnostic explanation):

The explanation model is defined as 
·      ( g ∈ G ) the constant g is a member of the set G.
·      G is a potentially interpretable model or explanation for families, such as linear models, decision trees, or falling rule lists. 
·      L ( f, g, πx ) is a measure of how unfaithful g (the over absence/presence of the interpretable components) is in approximating f in the locality defined by π
·      Ω(g)  - measures the complexity of g ∈ G 
o   For example, in decision trees, this would be the depth of the tree 
o   In linear models, this may be number of non-zero weights
·      L stands for fidelity functions 
The first step when using LIME is to minimize the locality-aware loss L ( f, g, πx )  without making assumptions about f, to allow the explainer the ability to explain any model – i.e keeping it model-agnostic.    

LIME Method

LIME 
Algorithm that can explain the predictions of any classifier or regressor by approximating it locally with an interpretable model.
SP-LIME
A method that uses a set of representative instances with explanations to build trust with the user via submodular optimization.
Comprehensive Evaluation
Using simulated and human subjects, we measure the impact of explanations on the user’s trust and associated tasks.
In the example of the article, non-experts using LIME are able to pick which classifier from a pair generalizes better in the real world. Thus, improving an untrustworthy classifier trained on 20 newsgroups, by doing feature engineering using LIME. 
Also, usage of neural networks on images help users understand why they shouldn’t trust a model. 


EXPLANATIONS
“Explaining a Prediction” means presenting textual or visual artifacts that provide an understanding of the relationship between the instance’s components and the model’s prediction. 
Explaining predictions is an important part of getting humans to trust and use ML effectively.
Development and Evaluation of a classification model normally consists of collecting annotated data, with a subset held out for automated evaluation. 
This “standard of practice” is not always ideal because looking at examples offers an alternative method to assess truth in the model, especially if the examples are explained. 
Models can go wrong several ways.
Data Leakage = unintentional leakage of signal into the training (and validation) data that would not appear when deployed
Dataset Shift = when training data is different from the test data
Explanations can be incredibly helpful in identifying what must be done to convert an untrustworthy model into a trustworthy one.
i.e., removing leaked data or changing the training data to avoid dataset shift. 

DESIRED CHARACTERISTICS FOR EXPLAINERS
Must be interpretable
I.e., provide qualitative understanding between the input variables and the response. 
Interpretability must take into account the user’s limitations. 
Local Fidelity
For an explanation to be meaningful, it must at least be locally faithful, i.e., must correspond to how the model behaves in the vicinity of the instance being predicted.
Local Fidelity does not imply Global Fidelity

LOCAL INTERPRETABLE MODEL-AGNOSTIC EXPLANATIONS
Goal: to identify an interpretable model over the interpretable representation that is locally faithful to the classifier

Fidelity-Interpretability Tradeoff
Sampling for Local Exploration
Sparse Linear Explanations

SUBMODULAR PICKS FOR EXPLAINING MODELS
 B = # of explanations/explanatory variables

N * d’ = explanation matrix W = local importance of the interpretable components for each instance

Variable x within an SP Algorithm is a vector


Algorithm 1: Sparse Linear Explanations using LIME
	Require: Classifier f, number of samples N
	Require: Instance x, and its interpretable version x’
	Require: Similarity kernel pie(x), Length of explanation K

Z = {}
For i (= {1, 2, 3, …., N} do
	z’ <= sample_around(x’)
	Z’ <= Z U (z’, f(zi), pie(x)(zi))
End for
W <= K-lasso(Z, K) with z’, as features, f(z) as target
Return w





SIMULATED USER EXPERIMENTS

CONCLUSION AND FURTHER STUDIES

To summarize, LIME is a method of explanation of a prediction or description of any machine learning model. It is crucial to data analysts to make their models explainable and transparent. Sometimes, we just get driven away by the results and forget about how we come up to those outputs. The questions are how we can understand a model and how can we be confident to trust the prediction of that model. LIME is highly recommended for model interpretability because of being local-faithful and model-independent or model-agnostic, just like the name of this technique. But we must stretch out that interpretable explanations and interpretable representation are different, and this leads to the fidelity-interpretability trade -off when we apply LIME to a model. The data representation should be understandable to human beings regardless of features. The example that we can mention here is : text classification task as a binary vector as output (present word/missing word), regardless of the number of input features. But LIME is local faithful, which means that the local explanations must fit the predictions that we obtain from the chosen model. The more features we feed to the local explanation, the more accurate are the outputs and will match closer to the original prediction but then the explanation will be more complex as there are more features which contribute to this explanation.
But LIME, as its name already described, explains models only locally. In order to understand a model globally, submodular pick or SP-LIME will be used. The method behind SP-LIME is to pick only the most important features from the dataset. The global importance of each feature is determined by the number of instances that each feature can explain. The more point values are explained, the more important is that feature. Unlike LIME, SP-LIME does not have redundant predictions and will provide a global view of the model to the users.
Finally, LIME/SP-LIME is a method applied to explain any machine learning system. This new domain opens the door to the explanation of variety studies such as speech, text, medical domains, recommendation systems….











Bibliography 
Ribeiro, M. T., Singh, S., & Guestrin, C. (2016, August 9). “Why Should I Trust You?” Explaining the Predictions of Any Classifier [PDF]. Seattle: University of Washington.

