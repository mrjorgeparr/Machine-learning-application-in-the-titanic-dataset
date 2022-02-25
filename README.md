# Machine-learning-application-in-the-titanic-dataset

This application solves the supervised learning task of predicting whether a passenger lives or survives on the titanic based on all the other information we are given on him.
The application uses decision trees, k-nearest neighbours, random forest and support vector machines with four different kernels: linear, polynomial, sigmoid and radial. For each of the classifiers we tune the hyperparameters to find the greatest accuracy achievable and then we choose the best performing one out of them, lastly the best accuracy is printed on screen, which comes from the random forest algorithm.

In order to make the application faster, with the help of the pracma, doParallel and foreach libraries we set it up so that parallel processing is done using the maximum number of cores that the device provides otherwise the application would be slowed greatly. Since all the computations done at iteration i are independent than those done at any iteration previous to it, there are no race conditions to be worried about.

The program calls a main function, from which subfunctions corresponding to each of the different classifiers are called, the accuracies obtained and for which values of the parameters they are obtained, then it compares to determine which is the best performing classifier and returns it.

Not all the parameters are tuned in every classifier, some are left with the default values. The most notable case is the random forest, in which only the two parameters are tuned, since adding more would in turn result in a great number of combinations, which in this particular algorithm would greatly slow computation since it uses the majority approach and considers hundreds of decision trees per iteration.
