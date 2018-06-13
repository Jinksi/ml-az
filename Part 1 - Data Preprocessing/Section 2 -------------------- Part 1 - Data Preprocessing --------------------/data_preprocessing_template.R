# Data Preprocessing

# Importing the dataset

dataset = read.csv('Data.csv')

# Missing data
dataset$Age = ifelse(is.na(dataset$Age),
                     # Ask R to remove missing values from mean calculation
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     # return existing value if exists
                     dataset$Age
                     )
dataset$Salary = ifelse(is.na(dataset$Salary),
                     # Ask R to remove missing values from mean calculation
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     # return existing value if exists
                     dataset$Salary
                    )
