# H2o.ai_DeepLearning

library(h2o)
h2o.init()

data <- h2o.importFile("http://h2o-public-test-data.s3.amazonaws.com/smalldata/airlines/allyears2k_headers.zip")

parts <- h2o.splitFrame(data, c(0.8,0.1), seed = 69)

train <- parts[[1]]
valid <- parts[[2]]

y <- "IsArrDelayed"
xAll <- setdiff(colnames(data), c("ArrDelay", "DepDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "IsDepDelayed", "IsArrDelayed", "ActualElapsedTime","Arrtime"))

m_def <- h2o.deeplearning(xAll, y, train, validation_frame = valid)

h2o.performance(m_def, valid = TRUE)

m_200_epochs <- h2o.deeplearning(xAll, y, train, validation_frame = valid, epochs =  200, stopping_rounds = 5, stopping_tolerance = 0, stopping_metric = "logloss")

h2o.performance(m_200_epochs, valid = TRUE)

plot(m_200_epochs)

h2o.scoreHistory(m_200_epochs)

m_200x200x200 <- h2o.deeplearning(xAll, y, train, validation_frame = valid, epochs =  200,hidden = c(200,200,200))

h2o.performance(m_200x200x200, valid = TRUE)

plot(m_200x200x200)
