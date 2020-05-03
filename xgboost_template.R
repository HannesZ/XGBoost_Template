
# XGBOOST OUT OF THE BOX:
# Regression:

# As prerequisit, we only need a dataframe "DT1" with our data.

# Given a set of explenatory variables among which are some categorical "CatVar1", "CatVar2", "CatVar3" and one numerical "NumVar1". Assume we wanted to fit the numerical variable "NumVarTarget":
Prediktoren <- c("NumVar1", "CatVar1", "CatVar2", "CatVar3")

cols <- union("NumVarTarget", Prediktoren)

# OPTIONAL: in some cases, it is necessary to have weights, so let's assume there is a column "NumVarWeight" with weights:
weight     <- "NumVarWeight"

# we use data.table, so make sure DT1 is data.table     :
setDT(DT1)

# we should restrict our data to records that don't include attributes we will not face in the inference-data. This could e.g include a certain insurance-coverage, that is no longer part of the portfolio. 
modell_filter <- which(DT1$CatVar2 != "Firlefanz")

# to avoid erros, let's :
TempXGB <- DT1[modell_filter ,.(NumVar1, CatVar1, CatVar2, CatVar3, NumVarTarget, NumVarWeight)]

# Alle nicht-numerischen Felder zu factor-Felder umwandeln und droplevels anwenden.
cat.cols <- names(TempXGB)[unlist(lapply(TempXGB, function(x){!is.numeric(x)}))]
TempXGB[,(cat.cols):=lapply(.SD, function(x){droplevels(as.factor(as.character(x)))}), .SDcols = cat.cols]

## Train-Test split:
anteilTest <- 0.2
ndat <- nrow(TempXGB)
set.seed(123)
trainSchnitz <-  sample(ndat, (1-anteilTest)*ndat)

expect_true(length(trainSchnitz) > 0, info = "Filter für die Auswahl des Trainingsdatensatzes scheint schlecht gewählt worden zu sein") # nicht immer so offensichtlich, wie hier

TrainDaten  <- TempXGB[trainSchnitz, cols, with = FALSE]
# optional:
TrainWeight  <- TempXGB[trainSchnitz, weight, with = FALSE]
TrainDaten[,(Prediktoren):=lapply(.SD, function(x){droplevels(as.factor((x)))}), .SDcols = Prediktoren]

TestDaten  <- TempXGB[-trainSchnitz, cols, with = FALSE]
# optional:
TestWeight  <- TempXGB[-trainSchnitz, weight, with = FALSE]
TestDaten[ ,(Prediktoren):=lapply(.SD, function(x){droplevels(as.factor(x))}), .SDcols = Prediktoren]

## Data.Frame mit Dummy-Variable:
DatenXGB   <- TempXGB[, cols, with = FALSE];rm("TempXGB");invisible(gc())
DatenXGB <- as.data.frame(DatenXGB)

library(mlr)
library(xgboost)

# 
# erklärende numerische Variablen kommen nicht hier rein, nur die Target-Variable und bei cols die erklärenden kagetorischen Varible:
DatenXGB <- createDummyFeatures(
  DatenXGB, target = "NumVarTarget",
  cols = c(
    "CatVar1",
    "CatVar2",
    "CatVar3"
  )
)

dummyfeatures <- as.vector(names(DatenXGB))

TrainDatenXGB <- DatenXGB[ trainSchnitz,]

TestDatenXGB  <- DatenXGB[-trainSchnitz,];rm("DatenXGB");invisible(gc())


trainTask <- makeRegrTask(data = TrainDatenXGB, target = "NumVarTarget", weights = TrainWeight) # weights optional
testTask  <- makeRegrTask(data = TestDatenXGB,  target = "NumVarTarget", weights = TestWeight) # weights optional

## Lerner-Objekt erstellen:
set.seed(321)
xgb_learner <- makeLearner(
  "regr.xgboost",
  predict.type = "response",
  par.vals = list(
    objective   = "reg:linear",
    eval_metric = "rmse",
    nrounds     = 300
    
  )
)

## Parameter tunen:
# Hier wird festgelegt, welche Parameter getuned werden. es gibt noch eine Unzahl weitere Parameter, siehe:
# https://xgboost.readthedocs.io/en/latest/parameter.html

xgb_params <- makeParamSet(
  # The number of trees in the model (each one built sequentially)
  makeIntegerParam("nrounds", lower = 300, upper = 1000),
  # number of splits in each tree
  makeIntegerParam("max_depth", lower = 2, upper = 10),
  # "shrinkage" - prevents overfitting
  makeNumericParam("eta", lower = .005, upper = .05),
  # # L2 regularization - prevents overfitting
  # makeNumericParam("lambda", lower = -1, upper = 0, trafo = function(x) 10^x),
  
  # using only a sub-sample on an iteration step
  makeNumericParam("subsample", lower = .3, upper = .8)
)


control <- makeTuneControlRandom(maxit = 1)
# Create a description of the resampling plan
resample_desc <- makeResampleDesc("CV", iters = 2)

library(parallelMap)
library(parallel)

anzahlCores <-  detectCores()-1

parallelStart(mode= "socket", cpus=anzahlCores)

tuned_params <- tuneParams(
  learner = xgb_learner,
  task = trainTask,
  resampling = resample_desc,
  par.set = xgb_params,
  control = control
)

## Erstellen eines Lerners mit getunten Parametern:
# Create a new model using tuned hyperparameters
xgb_tuned_learner <- setHyperPars(
  learner = xgb_learner,
  par.vals = tuned_params$x
)

cv_folds <- makeResampleDesc("CV", iters = 3) # 3 fold cross validation


## Modellanwendung:
xgb.model <- train(xgb_tuned_learner, trainTask)

parallelStop()

Prediktoren <- c("NumVar1", "CatVar1", "CatVar2", "CatVar3")
cols        <- c(Prediktoren, "NumVarTarget")

# wenn Ausprägungen kategorischer Variable Leerzeichen beinhalten, dann werden die bei Modellerzeugung durch Punkte ersetzt. Dies muss bei den Anwedungsdatensätzen ebenfalls erfolgen.
cat.cols <- names(DT2)[unlist(lapply(DT2, function(x){!is.numeric(x)}))]
DT2[,(cat.cols):=lapply(.SD, gsub, pat="\\s", rep=".", perl=T), .SDcols = cat.cols]

# kategorische Variablen im Modellierungsset müssen die gleichen Levels wie die entsprechenden Variablen im Modell haben:
mod.cols <- intersect(Prediktoren, cat.cols)


library(testthat)
for(col in mod.cols){
  expect_equal(levels(TempXGB[, get(!!col)]), gsub(paste0(col, "."), "", grep(!!paste0(col,"\\."), xgb.model$features, value=T)), # letzte Änderung !!paste0(col,"\\.") anstelle von !!col muss noch ausführlich getestet werden
               info = paste("Folgende Ausprägung ist nicht konsistent: ", setdiff(union(levels(TempXGB[, get(col)]), gsub(paste0(col, "."),"", grep(col, xgb.model$features, value=T))), intersect(levels(TempXGB[, get(col)]), gsub(paste0(col, "."),"", grep(col, xgb.model$features, value=T))))))
}

# Nach allfälliger Löschung der Datensätze mit unzulässigen level müssen die Level selber gelöscht werden:
DT2[,(mod.cols):=lapply(.SD, function(x){droplevels(as.factor(x))}), .SDcols = mod.cols]

# Dataframe mit der Target-Variable initialisiert auf 0 (notwendig!).
DT2_Forecast <- as.data.frame(DT2[,.(NumVar1, NumVarTarget=0, CatVar1, CatVar2, CatVar3)])

# Auch hier muss ein dummy-data.frame erstellt werden (Reihenfolge der kategorischen Variable scheint eine Rolle zu spielen!!!):
DT2_Dummy <- createDummyFeatures(
  DT2_Forecast, target = "NumVarTarget",
  cols = c(
    "CatVar1", 
    "CatVar2",
    "CatVar3"
  )
)

# Der Fit:
RegrTask <- makeRegrTask(data = DT2_Dummy, target = "NumVarTarget")
res_RegrTask <- predict(xgb.model, RegrTask)
DT2          <- cbind(DT2, res_RegrTask$data)
DT2[,c("id", "truth"):=NULL]
setnames(DT2, "response", "NumVarTarget.pred")

getDummyCols <- function(x, DT){
  cols <- NULL
  for(c in x){
    cols <- union(cols, grep(paste0("^", c, "\\..+$"), names(DT), value = T))
  }
  return(cols)
}

# Validierung:
#Dependencies in der richtigen Reihenfolge:
getFeatureImportance(freq.model)[1][[1]][order(getFeatureImportance(freq.model)[1][[1]], decreasing = T)]
