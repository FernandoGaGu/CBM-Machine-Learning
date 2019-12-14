# AUTHOR: Fernando García

# Install packages
library(devtools)
install_github('ramhiser/datamicroarray')

# Load dataset
library(datamicroarray)
data('gravier', package = 'datamicroarray')

#=============================================================
# ||||||||||||||||| DATA PRE-PROCESSING ||||||||||||||||||||||
#=============================================================

library(mlr)

data <- gravier$x; data[1:4,1:4] # Get data  
labels <- gravier$y; labels[1:4] # Get labels
data$label <- labels  # Append labels
# Create task
task <- makeClassifTask(id = "Breast_Cancer", data = data, target = "label", positive = "poor")
# Normalization (x - mean) / std
norm.task <- normalizeFeatures(task, method = "standardize")

# IMPORTANT NOTE: 
# It is an unbalanced problem, we have many more classes of one kind than another. 
# In this case we are interested in reducing the false negative rate.

#=============================================================
# ||||||||||||||||| NOM-PROBABILISTIC SUPER. |||||||||||||||||
#=============================================================
library(parallelMap)
options(wran = -1)

parallelStartSocket(4)
rdesc <- makeResampleDesc("CV", iters = 10, stratify = T)

# FILTERING ==================================================
# Univariate filtering
univariate.1.par.set <- makeParamSet(
  makeDiscreteParam("fw.method", 
                    values = c("FSelector_symmetrical.uncertainty", "kruskal.test")),
  makeDiscreteParam("fw.perc", 
                    values = c(0.05, 0.1, 0.2, 0.4, 0.6, 0.8)))

# Multivariate filtering
multivariate.1.par.set <- makeParamSet(
  makeDiscreteParam("fw.method", 
                    values = c("praznik_JMIM", "randomForest_importance")),
  makeDiscreteParam("fw.perc", 
                    values = c(0.05, 0.1, 0.2, 0.4, 0.6, 0.8)))


# ALGORITHM 1 ================================================
lrn.svm <- makeFilterWrapper("classif.ksvm", kernel = "rbfdot")

# Tuning stage
univariate.tuning <- tuneParams(learner = lrn.svm, 
                                task = norm.task,
                                par.set = univariate.1.par.set,
                                control = makeTuneControlGrid(),
                                resampling = rdesc,
                                measures = list(fnr, fpr, acc))

multivariate.tuning <- tuneParams(learner = lrn.svm,
                                  task = norm.task,
                                  par.set = multivariate.1.par.set,
                                  control = makeTuneControlGrid(),
                                  resampling = rdesc,
                                  measures = list(fnr, fpr, acc))

svm.wrapper.GA <- makeFeatSelWrapper(learner = makeLearner("classif.ksvm", kernel = "rbfdot"),
                                     resampling = rdesc,
                                     measures = list(fnr, fpr, acc),
                                     control = makeFeatSelControlGA(maxit = 10, 
                                                                    mutation.rate = 0.1, 
                                                                    crossover.rate = 0.5,
                                                                    mu = 100))  

lrn.svm.univariate <- makeFilterWrapper(learner = "classif.ksvm",
                                        kernel = "rbfdot",
                                        fw.method = univariate.tuning$x$fw.method,
                                        fw.perc = univariate.tuning$x$fw.perc)

lrn.svm.multivariate <- makeFilterWrapper(learner = "classif.ksvm",
                                          kernel = "rbfdot",
                                          fw.method = multivariate.tuning$x$fw.method,
                                          fw.perc =  multivariate.tuning$x$fw.perc)


# ALGORITHM 2 ===============================================
lrn.knn <- makeFilterWrapper("classif.kknn", kernel = "triangular")

# Tuning stage
univariate.tuning.2 <- tuneParams(learner = lrn.knn, 
                                  task = norm.task,
                                  par.set = univariate.1.par.set,
                                  control = makeTuneControlGrid(),
                                  resampling = rdesc,
                                  measures = list(fnr, fpr, acc))

multivariate.tuning.2 <- tuneParams(learner = lrn.knn,
                                    task = norm.task,
                                    par.set = multivariate.1.par.set,
                                    control = makeTuneControlGrid(),
                                    resampling = rdesc,
                                    measures = list(fnr, fpr, acc))


# Wrapper Genetic Algorithm
knn.wrapper.GA <- makeFeatSelWrapper(learner = "classif.kknn",
                                     resampling = rdesc,
                                     measures = list(fnr, fpr, acc),
                                     control = makeFeatSelControlGA(maxit = 10, 
                                                                    mutation.rate = 0.1, 
                                                                    crossover.rate = 0.5,
                                                                    mu = 100))  

lrn.knn.univariate <- makeFilterWrapper(learner = "classif.kknn", 
										kernel = "triangular",
                                        fw.method = univariate.tuning.2$x$fw.method,
                                        fw.perc = univariate.tuning.2$x$fw.perc)

lrn.knn.multivariate <- makeFilterWrapper(learner = "classif.kknn",
										  kernel = "triangular",
                                          fw.method = multivariate.tuning.2$x$fw.method,
                                          fw.perc =  multivariate.tuning.2$x$fw.perc)

# BENCHMARK =====================================================
# SetPredictType just to compute the ROC courve

lrn.svm <- makeLearner("classif.ksvm", kernel = "rbfdot", id = "SVM.Original.Features")
lrn.svm = setPredictType(lrn.svm, "prob")

lrn.svm.univariate$id <- "SVM.Univariate.kruskal.test.Perc.0.05"            # Best result: kruskal.test, Perc = 0.05
lrn.svm.univariate = setPredictType(lrn.svm.univariate, "prob")

lrn.svm.multivariate$id <- "SVM.Multivariate.RFimportance.Perc.0.05"        # Best result: randomForest, Perc = 0.05
lrn.svm.multivariate = setPredictType(lrn.svm.multivariate, "prob")

svm.wrapper.GA$id <- "SVM.Wrapper.GA"
svm.wrapper.GA = setPredictType(svm.wrapper.GA, "prob")

lrn.knn <- makeLearner("classif.kknn", kernel = "triangular", id = "KNN.Original.Features")        
lrn.knn = setPredictType(lrn.knn, "prob")

lrn.knn.univariate$id <- "KNN.Univariate.kruskal.test.Perc.0.05"           # Best result: kruskal.test, Perc = 0.05
lrn.knn.univariate = setPredictType(lrn.knn.univariate, "prob")

lrn.knn.multivariate$id <- "KNN.Multivariate.RFimportance.Perc.0.05"       # Best result: randomForest, Perc = 0.05
lrn.knn.multivariate = setPredictType(lrn.knn.multivariate, "prob")

knn.wrapper.GA$id <- "KNN.Wrapper.GA"
knn.wrapper.GA = setPredictType(knn.wrapper.GA, "prob")

lrns <- list(lrn.svm, lrn.svm.univariate, lrn.svm.multivariate, svm.wrapper.GA,
             lrn.knn, lrn.knn.univariate, lrn.knn.multivariate, knn.wrapper.GA)

bmr <- benchmark(lrns, norm.task, rdesc, measures = list(acc, fpr, fnr)); bmr
write.csv(bmr, "bmr_1.csv")  # Save the results

# ROC ==========================================================                                 
df <- generateThreshVsPerfData(bmr, measures = list(fpr, tpr, mmce))
plotROCCurves(df)

# HONEST EVALUATION FROM BEST CLASSIFIERS =======================
best.1 <- bmr$learners$SVM.Univariate.kruskal.test.Perc.0.05    # Best Support Vector Machine from the previous step

best.2 <- bmr$learners$KNN.Univariate.kruskal.test.Perc.0.05    # Best K-Nearest neighbor from the previous step

test.rdesc <- makeResampleDesc("RepCV", folds = 10, stratify = TRUE)   # Repeated CV 

res.1 <- resample(learner = best.1,
                  task = norm.task,
                  resampling = test.rdesc,
                  models = TRUE,
                  measures = list(acc, fpr, fnr))

res.2 <- resample(learner = best.2, 
                  task = norm.task,
                  resampling = test.rdesc,
                  models = TRUE,
                  measures = list(acc, fpr, fnr))

df <- generateThreshVsPerfData(list(res.1, res.2), measures = list(fpr, tpr, mmce))
plotROCCurves(df)

# ACCESS TO SELECTED FEATURES =================================
lrn.1.selected.features <- sapply(res.1$models, getFilteredFeatures)
table(lrn.1.selected.features)

lrn.2.selected.features <- sapply(res.2$models, getFilteredFeatures)
table(lrn.2.selected.features)

#=============================================================
# |||||||||||||||||||| PROBABILISTIC SUPER. ||||||||||||||||||
#=============================================================
# ALGORITHM 3 ================================================
lrn.lda <- makeFilterWrapper("classif.lda")
# Tuning stage
univariate.tuning.3 <- tuneParams(learner = lrn.lda, 
                                  task = norm.task,
                                  par.set = univariate.1.par.set,
                                  control = makeTuneControlGrid(),
                                  resampling = rdesc,
                                  measures = list(fnr, fpr, acc))

multivariate.tuning.3 <- tuneParams(learner = lrn.lda,
                                    task = norm.task,
                                    par.set = multivariate.1.par.set,
                                    control = makeTuneControlGrid(),
                                    resampling = rdesc,
                                    measures = list(fnr, fpr, acc))

lda.wrapper.GA <- makeFeatSelWrapper(learner = makeLearner("classif.lda", method = 'mle'),
                                     resampling = rdesc,
                                     measures = list(fnr, fpr, acc),
                                     control = makeFeatSelControlGA(maxit = 10, 
                                                                    mutation.rate = 0.1, 
                                                                    crossover.rate = 0.5,
                                                                    mu = 100))  

lrn.lda.univariate <- makeFilterWrapper(learner = "classif.lda",
                                        predict.type = "prob",
                                        fw.method = univariate.tuning.3$x$fw.method,
                                        fw.perc = univariate.tuning.3$x$fw.perc)

lrn.lda.multivariate <- makeFilterWrapper(learner = "classif.lda",
                                          predict.type = "prob",
                                          fw.method = multivariate.tuning.3$x$fw.method,
                                          fw.perc =  multivariate.tuning.3$x$fw.perc)



# ALGORITHM 4 ===============================================
lrn.nab <- makeFilterWrapper("classif.naiveBayes",
                             predict.type = "prob")
# Tuning stage
univariate.tuning.4 <- tuneParams(learner = lrn.nab, 
                                  task = norm.task,
                                  par.set = univariate.1.par.set,
                                  control = makeTuneControlGrid(),
                                  resampling = rdesc,
                                  measures = list(fnr, fpr, acc))

multivariate.tuning.4 <- tuneParams(learner = lrn.nab,
                                    task = norm.task,
                                    par.set = multivariate.1.par.set,
                                    control = makeTuneControlGrid(),
                                    resampling = rdesc,
                                    measures = list(fnr, fpr, acc))
# Wrapper Genetic Algorithm
nab.wrapper.GA <- makeFeatSelWrapper(learner = makeLearner("classif.naiveBayes"),
                                     resampling = rdesc,
                                     measures = list(fnr, fpr, acc),
                                     control = makeFeatSelControlGA(maxit = 10, 
                                                                    mutation.rate = 0.1, 
                                                                    crossover.rate = 0.5,
                                                                    mu = 100))  

lrn.nab.univariate <- makeFilterWrapper(learner = "classif.naiveBayes",
                                        fw.method = univariate.tuning.4$x$fw.method,
                                        fw.perc = univariate.tuning.4$x$fw.perc)

lrn.nab.multivariate <- makeFilterWrapper(learner = "classif.naiveBayes",
                                          fw.method = multivariate.tuning.4$x$fw.method,
                                          fw.perc =  multivariate.tuning.4$x$fw.perc)


# BENCHMARK =====================================================   
# (I use setPredictType because I have had problems to compute the ROC courve 
#  using predict.type = "prob" in the classifier definition)

lrn.lda <- makeLearner("classif.lda", id = "LDA.Original.Features")
lrn.lda = setPredictType(lrn.lda, "prob")
lrn.lda.univariate$id <- "LDA.Univariate.Kruskal.test.Perc.0.2"                    # Best: Kruskal.test, Perc = 0.2
lrn.lda.univariate = setPredictType(lrn.lda.univariate, "prob")
lrn.lda.multivariate$id <- "LDA.Multivariate.RFimportance.Perc.0.2"     # Best: RandomForestImportance, Perc = 0.2
lrn.lda.multivariate = setPredictType(lrn.lda.multivariate, "prob")
lda.wrapper.GA$id <- "LDA.Wrapper.GA"
lda.wrapper.GA = setPredictType(lda.wrapper.GA, "prob")

lrn.nab <- makeLearner("classif.naiveBayes", predict.type = "prob", id = "NAB.Original.Features")
lrn.nab.univariate$id <- "NAB.Univariate.Sym.Unc.Perc.0.05"              # Sym.Unc, Perc = 0.05
lrn.nab.univariate = setPredictType(lrn.nab.univariate, "prob")
lrn.nab.multivariate$id <- "NAB.Multivariate.JMIM.Perc.0.05"               # praznik_JMIM, Perc = 0.05
lrn.nab.multivariate = setPredictType(lrn.nab.multivariate, "prob")
nab.wrapper.GA$id <- "NAB.Wrapper.GA"
nab.wrapper.GA = setPredictType(nab.wrapper.GA, "prob")

lrns.2 <- list(lrn.lda, lrn.lda.univariate, lrn.lda.multivariate, lda.wrapper.GA,
               lrn.nab, lrn.nab.univariate, lrn.nab.multivariate, nab.wrapper.GA)

bmr.2 <- benchmark(lrns.2, norm.task, rdesc, measures = list(acc, fpr, fnr)); bmr.2

write.csv(bmr.2, "bmr_2.csv")  # Save the results

# ROC ALL RESULTS ================================================
df <- generateThreshVsPerfData(bmr.2, measures = list(fpr, tpr, mmce))
plotROCCurves(df)

# HONEST EVALUATION FROM BEST CLASSIFIERS =======================
best.3 <- bmr.2$learners$LDA.Univariate.Kruskal.test.Perc.0.2    # Best Linear Discriminant classifier
best.4 <- bmr.2$learners$NAB.Multivariate.JMIM.Perc.0.05    # Best Naive Bayes classifier

test.rdesc <- makeResampleDesc("RepCV", folds = 10, stratify = TRUE)   # Repeated CV 

res.3 <- resample(learner = best.3,
                  task = norm.task,
                  resampling = test.rdesc,
                  models = TRUE,
                  measures = list(acc, fpr, fnr))

res.4 <- resample(learner = best.4, 
                  task = norm.task,
                  resampling = test.rdesc,
                  models = TRUE,
                  measures = list(acc, fpr, fnr))

df <- generateThreshVsPerfData(list(res.3, res.4), measures = list(fpr, tpr, mmce))
plotROCCurves(df)


# ACCESS TO SELECTED FEATURES =================================
lrn.3.selected.features <- sapply(res.3$models, getFilteredFeatures)
table(lrn.3.selected.features)

lrn.4.selected.features <- sapply(res.4$models, getFilteredFeatures)
table(lrn.4.selected.features)

# MMCE FOR PROBABILISTIC CLASSIFIERS ===========================
# LDA
d <- generateThreshVsPerfData(res.3, measures = list(mmce))
plotThreshVsPerf(d,
                 pretty.names = TRUE)

# NAB
d <- generateThreshVsPerfData(res.4, measures = list(mmce))
plotThreshVsPerf(d,
                 pretty.names = TRUE)

# ROC FOR ALL CLASSIFIERS ======================================
df <- generateThreshVsPerfData(list(res.1, res.2, res.3, res.4), measures = list(fpr, tpr, mmce))
plotROCCurves(df)

parallelStop()

#===========================================================
#||||||||||||||||||||||| ADABOOST ||||||||||||||||||||||||||
#===========================================================
parallelStartSocket(4)

lrn.ada <- makeFilterWrapper("classif.ada", 
                             fw.method = "kruskal.test",
                             predict.type = "prob")

rdesc <- makeResampleDesc("CV", iters = 5, stratify = T)

par.set <- makeParamSet(
  makeDiscreteParam("fw.abs", values = c(25, 50, 100, 150, 300)))

tuned.model <- tuneParams(learner = lrn.ada,
                          task = norm.task,
                          par.set = par.set,
                          control = makeTuneControlGrid(),
                          resampling = rdesc,
                          measures = list(fnr, fpr, acc))
# Take the best params
best.ada <- makeFilterWrapper("classif.ada",
                              fw.method = "kruskal.test",
                              fw.abs = tuned.model$x$fw.abs)

# ROC =========================================================
lrn.ada <- makeLearner("classif.ada", predict.type = "prob", id = "ADA.Default.Original.Features")
best.ada <- setPredictType(best.ada, "prob")
best.ada$id <- "ADA.Tuned.Selected.Features"   

rdesc.2 <- makeResampleDesc("RepCV", folds = 5, stratify = TRUE)

res.3 <- resample(learner = best.ada,
                  task = norm.task, 
                  resampling = rdesc.2,
                  measures = list(acc, fnr, fpr)); res.3

res.4 <- resample(learner = lrn.ada,
                  task = norm.task, 
                  resampling = rdesc.2,
                  measures = list(acc, fnr, fpr)); res.4

df <- generateThreshVsPerfData(list(res.1, res.2), measures = list(fpr, tpr, mmce))
plotROCCurves(df)


# LEARNING COURVE =============================================
r = generateLearningCurveData(
  learners = list(lrn.ada, best.ada),
  task = norm.task,
  percs = seq(0.1, 1, by = 0.2),
  measures = list(tp, fp, tn, fn),
  resampling = makeResampleDesc(method = "CV", iters = 5, stratify = T))
plotLearningCurve(r)

parallelStop()

#===========================================================
#|||||||||||||||||||| RANDOM FOREST ||||||||||||||||||||||||
#===========================================================
parallelStartSocket(4)

lrn.rf <- makeFilterWrapper("classif.randomForest", predict.type = "prob", fw.method = "kruskal.test")

par.set <- makeParamSet(
  makeDiscreteParam("fw.abs", values = c(20, 30, 50, 100)),
  makeDiscreteParam("ntree", values = c(350, 400, 450)),
  makeDiscreteParam("nodesize", values = c(1, 5, 10, 20, 40, 80)))

rdesc <- makeResampleDesc("CV", iters = 5, stratify = T)

tuned.model <- tuneParams(learner = lrn.rf,
                          task = norm.task,
                          par.set = par.set,
                          control = makeTuneControlGrid(),
                          resampling = rdesc,
                          measures = list(fnr, fpr, acc))

# Take the best params
best.rf<- makeFilterWrapper("classif.randomForest",
                            fw.method = "kruskal.test",
                            fw.abs = tuned.model$x$fw.abs,
                            ntree = tuned.model$x$ntree,
                            nodesize = tuned.model$x$nodesize)


# ROC =========================================================
lrn.rf <- makeLearner("classif.randomForest", predict.type = "prob", id = "RF.Default.Original.Features")
best.rf <- setPredictType(best.rf, "prob")
best.rf$id <- "RF.Tuned.Selected.Features"   

res.1 <- resample(learner = best.rf,
                  task = norm.task, 
                  resampling = rdesc.2,
                  measures = list(acc, fnr, fpr)); res.1

res.2 <- resample(learner = lrn.rf,
                  task = norm.task, 
                  resampling = rdesc.2,
                  measures = list(acc, fnr, fpr)); res.2

df <- generateThreshVsPerfData(list(res.1, res.2, res.3, res.4), measures = list(fpr, tpr, mmce))
plotROCCurves(df)

# LEARNING COURVE =============================================
r = generateLearningCurveData(
  learners = list(lrn.rf, best.rf),
  task = norm.task,
  percs = seq(0.1, 1, by = 0.2),
  measures = list(tp, fp, tn, fn),
  resampling = makeResampleDesc(method = "CV", iters = 5, stratify = T))
plotLearningCurve(r)

parallelStop()


# ======================================================
# ||||||||||||| BOXPLOT NO PROBABILISTIC |||||||||||||||
# ======================================================
library(ggplot2)

# LOAD DATA ============================================
data <- read.table('bmr_1.csv', sep = ",", header = TRUE, row.names = 1) # Rad data
data[,1:4]
# Replace long names for shortcut forms
data <- data.frame(lapply(data, function(x) {gsub("SVM.Original.Features", "SVM.OF", x)}))
data <- data.frame(lapply(data, function(x) {gsub("SVM.Univariate.kruskal.test.Perc.0.05", "SVM.Uni", x)}))
data <- data.frame(lapply(data, function(x) {gsub("SVM.Multivariate.RFimportance.Perc.0.05", "SVM.Mul", x)}))
data <- data.frame(lapply(data, function(x) {gsub("SVM.Wrapper.GA", "SVM.Wrap", x)}))
data <- data.frame(lapply(data, function(x) {gsub("KNN.Original.Features", "KNN.OF", x)}))
data <- data.frame(lapply(data, function(x) {gsub("KNN.Univariate.kruskal.test.Perc.0.05", "KNN.Uni", x)}))
data <- data.frame(lapply(data, function(x) {gsub("KNN.Multivariate.RFimportance.Perc.0.05", "KNN.Mul", x)}))
data <- data.frame(lapply(data, function(x) {gsub("KNN.Wrapper.GA", "KNN.Wrap", x)}))
data[,1:4]

# Convert number from character to numeric
data$acc <- sapply(sapply(data$acc, as.character), as.numeric)
data$fpr <- sapply(sapply(data$fpr, as.character), as.numeric)
data$fnr <- sapply(sapply(data$fnr, as.character), as.numeric)

data[1:4,]

accuracy <- data[2:4]   # Accuracy data
# To color the boxplot
ALGORITMO <- c(rep("SVM", 20), rep("KNN", 20))
accuracy$ALGORITMO <- ALGORITMO
accuracy$iter <- NULL
accuracy[,]

# BOXPLOT ACCURACY ===================================
plot <- ggplot(accuracy, aes(x = learner.id, y = acc, fill = ALGORITMO)) + 
  scale_fill_manual(values=c("#000000", "#D8D8D8"), name = "Clasificador") +  # To select class color
  geom_boxplot(alpha = 0.5, colour = "black") + 
  scale_x_discrete(name = "Algoritmo + Selección de características") +
  scale_y_continuous(name = "Precisión 5-CV", breaks = round(seq(0.5, 1, by = 0.1), 2),
                     limits = c(0.5,1)) +                                # 5 -CV
  ggtitle("Aprendizaje Supervisado no Probabilístico") +
  theme(plot.title = element_text(hjust = 0.6, size = 20, family = 'Times New Roman'),
        panel.border = element_blank(),
        panel.grid.minor = element_line(colour = "grey80"),
        panel.background = element_rect(fill = "white", colour = "white"),
        axis.line = element_line(colour = "black")) 
plot 



# Get the mean for fnr and fpr
# SVM
svm.of.fpr <- mean(data$fpr[data$learner.id == "SVM.OF"]); svm.of.fpr
svm.of.fnr <- mean(data$fnr[data$learner.id == "SVM.OF"]); svm.of.fnr

svm.uni.fpr <- mean(data$fpr[data$learner.id == "SVM.Uni"]); svm.uni.fpr
svm.uni.fnr <- mean(data$fnr[data$learner.id == "SVM.Uni"]); svm.uni.fnr

svm.mul.fpr <- mean(data$fpr[data$learner.id == "SVM.Mul"]); svm.mul.fpr
svm.mul.fnr <- mean(data$fnr[data$learner.id == "SVM.Mul"]); svm.mul.fnr

svm.wrap.fpr <- mean(data$fpr[data$learner.id == "SVM.Wrap"]); svm.wrap.fpr
svm.wrap.fnr<- mean(data$fnr[data$learner.id == "SVM.Wrap"]); svm.wrap.fnr
# KNN
knn.of.fpr <- mean(data$fpr[data$learner.id == "KNN.OF"]); knn.of.fpr
knn.of.fnr <- mean(data$fnr[data$learner.id == "KNN.OF"]); knn.of.fnr

knn.uni.fpr <- mean(data$fpr[data$learner.id == "KNN.Uni"]); knn.uni.fpr
knn.uni.fnr <- mean(data$fnr[data$learner.id == "KNN.Uni"]); knn.uni.fnr

knn.mul.fpr <- mean(data$fpr[data$learner.id == "KNN.Mul"]); knn.mul.fpr
knn.mul.fnr <- mean(data$fnr[data$learner.id == "KNN.Mul"]); knn.mul.fnr

knn.wrap.fpr <- mean(data$fpr[data$learner.id == "KNN.Wrap"]); knn.wrap.fpr
knn.wrap.fnr<- mean(data$fnr[data$learner.id == "KNN.Wrap"]); knn.wrap.fnr

lrn.id <- c(rep("SVM.OF", 2), rep("SVM.Uni", 2), rep("SVM.Mul", 2),rep("SVM.Wrap", 2),
            rep("KNN.OF", 2),rep("KNN.Uni", 2),rep("KNN.Mul", 2), rep("KNN.Wrap", 2))

label <- rep(c('fpr', 'fnr'), 8)

values <- c(svm.of.fpr, svm.of.fnr, svm.uni.fpr, svm.uni.fnr, svm.mul.fpr, svm.mul.fnr, svm.wrap.fpr, svm.wrap.fnr,
            knn.of.fpr, knn.of.fnr,knn.uni.fpr, knn.uni.fnr, knn.mul.fpr, knn.mul.fnr, knn.wrap.fpr, knn.wrap.fnr)

f.positive.negative <- data.frame(lrn.id, label, values)

# ======================================================
# ||||||||||||||| BOXPLOT PROBABILISTIC ||||||||||||||||
# ======================================================
# LOAD DATA ============================================     
data <- read.table('bmr_2.csv', sep = ",", header = TRUE, row.names = 1) # Read data
data[,1:4]
# Replace long names for shortcut forms
data <- data.frame(lapply(data, function(x) {gsub("LDA.Original.Features", "LDA.OF", x)}))
data <- data.frame(lapply(data, function(x) {gsub("LDA.Univariate.Kruskal.test.Perc.0.2", "LDA.Uni", x)}))
data <- data.frame(lapply(data, function(x) {gsub("LDA.Multivariate.RFimportance.Perc.0.2", "LDA.Mul", x)}))
data <- data.frame(lapply(data, function(x) {gsub("LDA.Wrapper.GA", "LDA.Wrap", x)}))
data <- data.frame(lapply(data, function(x) {gsub("NAB.Original.Features", "NAB.OF", x)}))
data <- data.frame(lapply(data, function(x) {gsub("NAB.Univariate.Sym.Unc.Perc.0.05", "NAB.Uni", x)}))
data <- data.frame(lapply(data, function(x) {gsub("NAB.Multivariate.JMIM.Perc.0.05", "NAB.Mul", x)}))
data <- data.frame(lapply(data, function(x) {gsub("NAB.Wrapper.GA", "NAB.Wrap", x)}))
data[,1:4]

# Convert number from character to numeric
data$acc <- sapply(sapply(data$acc, as.character), as.numeric)
data$fpr <- sapply(sapply(data$fpr, as.character), as.numeric)
data$fnr <- sapply(sapply(data$fnr, as.character), as.numeric)

data[1:4,]

accuracy <- data[2:4]   # Accuracy data
# To color the boxplot
ALGORITMO <- c(rep("LDA", 20), rep("NAB", 20))
accuracy$ALGORITMO <- ALGORITMO
accuracy$iter <- NULL
accuracy[,]

# BOXPLOT ACCURACY ===================================
accuracy
plot <- ggplot(accuracy, aes(x = learner.id, y = acc, fill = ALGORITMO)) + 
  scale_fill_manual(values=c("#000000", "#D8D8D8"), name = "Clasificador") +  # To select class color
  geom_boxplot(alpha = 0.5, colour = "black") + 
  scale_x_discrete(name = "Algoritmo + Selección de características") +
  scale_y_continuous(name = "Precisión 5-CV", breaks = round(seq(0.5, 1, by = 0.1), 2),
                     limits = c(0.5,1)) +                                # 5 -CV
  ggtitle("Aprendizaje Supervisado  Probabilístico") +
  theme(plot.title = element_text(hjust = 0.6, size = 20, family = 'Times New Roman'),
        panel.border = element_blank(),
        panel.grid.minor = element_line(colour = "grey80"),
        panel.background = element_rect(fill = "white", colour = "white"),
        axis.line = element_line(colour = "black")) 
plot 

# SAVE fpr and fnr VALUES =====================================
lda.of.fpr <- mean(data$fpr[data$learner.id == "LDA.OF"]); lda.of.fpr
lda.of.fnr <- mean(data$fnr[data$learner.id == "LDA.OF"]); lda.of.fnr

lda.uni.fpr <- mean(data$fpr[data$learner.id == "LDA.Uni"]); lda.uni.fpr
lda.uni.fnr <- mean(data$fnr[data$learner.id == "LDA.Uni"]); lda.uni.fnr

lda.mul.fpr <- mean(data$fpr[data$learner.id == "LDA.Mul"]); lda.mul.fpr
lda.mul.fnr <- mean(data$fnr[data$learner.id == "LDA.Mul"]); lda.mul.fnr

lda.wrap.fpr <- mean(data$fpr[data$learner.id == "LDA.Wrap"]); lda.wrap.fpr
lda.wrap.fnr<- mean(data$fnr[data$learner.id == "LDA.Wrap"]); lda.wrap.fnr
# NAB
nab.of.fpr <- mean(data$fpr[data$learner.id == "NAB.OF"]); nab.of.fpr
nab.of.fnr <- mean(data$fnr[data$learner.id == "NAB.OF"]); nab.of.fnr

nab.uni.fpr <- mean(data$fpr[data$learner.id == "NAB.Uni"]); nab.uni.fpr
nab.uni.fnr <- mean(data$fnr[data$learner.id == "NAB.Uni"]); nab.uni.fnr

nab.mul.fpr <- mean(data$fpr[data$learner.id == "NAB.Mul"]); nab.mul.fpr
nab.mul.fnr <- mean(data$fnr[data$learner.id == "NAB.Mul"]); nab.mul.fnr

nab.wrap.fpr <- mean(data$fpr[data$learner.id == "NAB.Wrap"]); nab.wrap.fpr
nab.wrap.fnr<- mean(data$fnr[data$learner.id == "NAB.Wrap"]); nab.wrap.fnr

lrn.id.2 <- c(rep("LDA.OF", 2), rep("LDA.Uni", 2), rep("LDA.Mul", 2),rep("LDA.Wrap", 2),
              rep("NAB.OF", 2),rep("NAB.Uni", 2),rep("NAB.Mul", 2), rep("NAB.Wrap", 2))

label.2 <- rep(c('fpr', 'fnr'), 8)

values.2 <- c(lda.of.fpr, lda.of.fnr, lda.uni.fpr, lda.uni.fnr, lda.mul.fpr, lda.mul.fnr, lda.wrap.fpr, lda.wrap.fnr,
              nab.of.fpr, nab.of.fnr,nab.uni.fpr, nab.uni.fnr, nab.mul.fpr, nab.mul.fnr, nab.wrap.fpr, nab.wrap.fnr)

f.positive.negative.2 <- data.frame(lrn.id.2, label.2, values.2)
colnames(f.positive.negative.2) <- c("lrn.id", "label", "values")

# ======================================================
# ||||||||||||||||||||| BARCHART |||||||||||||||||||||||
# ======================================================
library(reshape2)

f.positive.negative <- rbind(f.positive.negative, f.positive.negative.2)   # Merge all fnr and fpr
dim(f.positive.negative)

ggplot(f.positive.negative, aes(x = lrn.id, y = values)) + 
  scale_fill_manual(values=c("#000000", "#666E67"), name = "Etiqueta") +  # To select class color
  geom_bar(aes(fill = label), stat = "identity",position = "dodge", alpha = 0.8) +
  scale_x_discrete(name = "Media de de falsos positivos y negativos") +
  scale_y_continuous(name = "Tasa de fpr o fnr", breaks = round(seq(0,1, by = 0.1), 2),
                     limits = c(0,1)) + 
  ggtitle("Tasas de falsos positivos y falsos negativos") + 
  theme(plot.title = element_text(hjust = 0.5, size = 20, family = 'Times New Roman'),
        panel.border = element_blank(),
        panel.grid.minor = element_line(colour = "grey80"),
        panel.background = element_rect(fill = "white", colour = "white"),
        
        axis.line = element_line(colour = "black")) 

#=============================================================
# |||||||||||||||||||||||| RELIEF ||||||||||||||||||||||||||||
#=============================================================
library(mlr)
library(dplyr)

data <- gravier$x; data[1:4,1:4] # Get data  

labels <- gravier$y; labels[1:4] # Get labels

get_top <-function(x, n){
  best <- x$data
  best <- best[order(-best$value), ]
  best <- best[1:n, ]
  return(best)
}

# SVM
task <- makeClassifTask(data = data, target = 'labels', positive = 'poor')
norm.task <- normalizeFeatures(task, method = "standardize")

fv <- generateFilterValuesData(norm.task, method = c("FSelector_relief"))   # Multivariate (take time)
top.relief <- get_top(fv, 100)
write.csv(top.relief, file = "Top_100_RELIEF.csv")

#=============================================================
# ||||||||||||||||||| RELEVANT.RELIEF ||||||||||||||||||||||||
#=============================================================
data <- gravier$x; data[1:4,1:4] # Get data  

labels <- gravier$y; labels[1:4] # Get labels
## LOAD RELIEF DATA
top.relief <- read.table("Top_100_RELIEF.csv", sep = ",", header = TRUE, row.names = 1)
data <- data[, top.relief$name]
data$labels <- labels
dim(data)

task <- makeClassifTask(data = data, target = 'labels', positive = 'poor')
norm.task <- normalizeFeatures(task, method = "standardize")

# SEQUENTIAL + SVM ==================================================
svm.ga <-  makeFeatSelWrapper(learner = makeLearner("classif.ksvm", kernel = "rbfdot"),
                              resampling = makeResampleDesc("CV", iters = 5, stratify = T),
                              measures = list(acc, fnr, fpr),
                              control = makeFeatSelControlSequential(maxit = 10,
                                                                     method = 'sfs',
                                                                     max.features = 15
                              ))  
ctrl = makeFeatSelControlSequential(maxit = 200,
                                    method = 'sfs',
                                    max.features = 15)

sfeats <- selectFeatures(learner = makeLearner("classif.ksvm", kernel = "rbfdot"),
                         task = norm.task,
                         control = ctrl,
                         resampling = makeResampleDesc("CV", iters = 10, stratify = T),
                         measures = list(acc, fnr, fpr))
sfeats
# FeatSel result:
# Features (5): g2F06, g2H06, g1H05, g3A08, g1int7
# acc.test.mean=0.7969771,fnr.test.mean=0.4100000,fpr.test.mean=0.0984848

#=============================================================
# ||||||||||||||| HIERARCHICAL CLSUTERING ||||||||||||||||||||
#=============================================================
library(mclust)
library(factoextra)

significant.genes <- c("g2F06", "g2H06", "g1H05", "g3A08", "g1int7")
data <- gravier$x
data <- data[, significant.genes]
data <- as.data.frame(scale(as.matrix(data), center = T, scale = T))
data$labels <- gravier$y
dim(data)
write.csv(data, "no_supervised_genes.csv")

data.all <- data
row.names <- paste(data.all$labels, rownames(data.all), sep = '.'); row.names[1:4]
labels = data.all$labels
rownames(data.all) <- row.names
data.all$labels <- NULL

dist_matrix <- dist(data.all, method = 'euclidean')
hclust_ward <- hclust(dist_matrix, method = 'ward.D')
cut_tree <- cutree(hclust_ward, k = 4)
plot(hclust_ward, main = "Cluster Herárquico")

rect.hclust(hclust_ward, k = 4, border = 2:10)
print(table(labels, cut_tree))

fviz_cluster(list(data = data.all, cluster = cut_tree), 
             geom = c("point"),
             ggtheme = theme_bw())

#=============================================================
# |||||||||||||||||||||| K-MEANS + PCA||||||||||||||||||||||||
#=============================================================
library(gridExtra)
# FIND THE OPTIMAL VALUES
p1 <- fviz_nbclust(data.all, kmeans, method = "wss")
p2 <- fviz_nbclust(data.all, kmeans, method = "silhouette")

km.2 <- kmeans(data.all, 2, nstart = 10)
p3 <- fviz_cluster(km.2, data.all, ellipse.type = "norm",
                   ellipse.level = 0.95,
                   ellipse.alpha = 0,
                   main = "K-Means (k = 2)",
                   geom = c("point"), 
                   ggtheme =   theme_bw(base_family = "Times New Roman",
                                        base_size = 15)
)
km.2$cluster[km.2$cluster == 1]   # 94 good   # 30 poor
km.2$cluster[km.2$cluster == 2]   # 17 good  # 27 poor


km.4 <- kmeans(data.all, 4, nstart = 10)
p4 <- fviz_cluster(km.4, data.all, ellipse.type = "norm",
                   ellipse.level = 0.95,
                   ellipse.alpha = 0,
                   main = "K-Means (k = 4)",
                   geom = c("point"), 
                   ggtheme =   theme_bw(base_family = "Times New Roman",
                                        base_size = 15)
)
km.4$cluster[km.4$cluster == 1]   # 9 good   # 11 poor
km.4$cluster[km.4$cluster == 2]   # 14 good  # 17 poor
km.4$cluster[km.4$cluster == 3]   # 21 good  # 15 poor
km.4$cluster[km.4$cluster == 4]   # 67 good  # 14 poor

grid.arrange(p1, p2, p3, p4, nrow = 2)

#=============================================================
# ||||||||||||||||||| GAUSSIAN MIXTURES ||||||||||||||||||||||
#=============================================================
BIC <- mclustBIC(data.all)
plot(BIC)
summary(BIC)
mclustModelNames("VEE")

mod1 <- Mclust(data.all, x = BIC)
plot(mod1, what = "classification")
table(labels, mod1$classification)

#=============================================================
# |||||||||||||||||||||||| HEATMAP |||||||||||||||||||||||||||
#=============================================================
library(matlib)
library(maptools)
library(mlr)
library(RColorBrewer)

data.norm <- as.matrix(scale(data[,1:5], center = TRUE, scale = TRUE))  # center = TRUE (x - mean)/std // center = FALSE x/std
rownames(data.norm) <- labels
newcolors<-colorRampPalette(colors=c("red","black","green"))(256)
heatmap(as.matrix(data.all[,1:5]), col=newcolors )

cluster.1 <- rownames(as.data.frame(km.2$cluster[km.2$cluster == 1]))    # 94 good   # 30 poor
cluster.2 <- rownames(as.data.frame(km.2$cluster[km.2$cluster == 2]))    # 17 good  # 27 poor

# Mean per column
cluster.1 <- data.all[cluster.1,]
cluster.1.mean <- lapply(cluster.1, mean); cluster.1.mean
cluster.1.sd <- lapply(cluster.1, sd); cluster.1.sd
hist(as.matrix(cluster.1$g2F06))
hist(as.matrix(cluster.1$g2H06))
hist(as.matrix(cluster.1$g1H05))
hist(as.matrix(cluster.1$g3A08))
hist(as.matrix(cluster.1$g1int7))

cluster.2 <- data.all[cluster.2,]
cluster.2.mean <- lapply(cluster.2, mean); cluster.1.mean
cluster.2.sd <- lapply(cluster.2, sd); cluster.2.sd
hist(as.matrix(cluster.2$g2F06))
hist(as.matrix(cluster.2$g2H06))
hist(as.matrix(cluster.2$g1H05))
hist(as.matrix(cluster.2$g3A08))
hist(as.matrix(cluster.2$g1int7))

