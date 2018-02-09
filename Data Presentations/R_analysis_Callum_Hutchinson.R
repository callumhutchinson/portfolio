library(tidyverse)
library(readr)
#data analysis
HR_comma_sep_1_ = read_csv("Downloads/HR_comma_sep (1).csv")
data = HR_comma_sep_1_
data = rename(data,department=sales)
data = as_tibble(data)
data$left = as_tibble(data$left)
theme_set(theme_minimal())
min(data$last_evaluation)
max(data$last_evaluation)
quantile(data$last_evaluation,0.8)
data = mutate(data,high_performer=ifelse(last_evaluation>0.89,1,0))
data$high_performer = as_tibble(data$high_performer)
hp_data = filter(data,high_performer==1)
non_hp_data = filter(data,high_performer==0)
ggplot(hp_data) + geom_bar(mapping=aes(x=left,y=(..count..)/sum(..count..)),fill="steelblue") + scale_x_continuous(breaks=c(No=0,Yes=1)) + xlab("Left") + ylab("Percentage") + scale_y_continuous(breaks=seq(0,0.7,0.1))
count(hp_data,left)
count(non_hp_data,left)
ggplot() + geom_freqpoly(mapping=aes(x=hp_data$satisfaction_level,y=..density..,color="High Performers")) + geom_freqpoly(mapping=aes(x=non_hp_data$satisfaction_level,y=..density..,color="Not")) + xlab("Satisfaction Level") + ylab("Density")
ggplot() + geom_freqpoly(mapping=aes(x=hp_data$average_montly_hours,y=..density..,color="High Performers")) + geom_freqpoly(mapping=aes(x=non_hp_data$average_montly_hours,y=..density..,color="Not")) + xlab("Average Monthly Hours") + ylab("Density")
ggplot() + geom_boxplot(mapping=aes(x=data$high_performer, y=data$time_spend_company),color="steelblue") + xlab("High Performer Status") + ylab("Years at the Company") + coord_flip() + scale_x_discrete(breaks=c(No=0,Yes=1))
ggplot(hp_data) + geom_bar(mapping=aes(x=Work_accident,y=(..count..)/sum(..count..),fill="lightcoral")) + scale_x_continuous(breaks=c(No=0,Yes=1)) + scale_y_continuous(breaks=seq(0,0.9,0.1)) + xlab("Involed in a workplace accident?") + ylab("Percentage")
ggplot(non_hp_data) + geom_bar(mapping=aes(x=Work_accident,y=(..count..)/sum(..count..)),fill="steelblue") + scale_x_continuous(breaks=c(No=0,Yes=1)) + scale_y_continuous(breaks=seq(0,0.9,0.1)) + xlab("Involed in a workplace accident?") + ylab("Percentage")
t.test(hp_data$promotion_last_5years, non_hp_data$promotion_last_5years, "g", 0, FALSE, TRUE, 0.95)
count(hp_data,department) %>% ggplot() + geom_bar(mapping=aes(x=reorder(department,n),y=n),stat="identity",fill="lightcoral") + coord_flip() + xlab("Department") + ylab("Count")
dist_jobs = count(data,department) %>% mutate(num_hp=count(hp_data,department)$n) %>% mutate(num_non_hp=count(non_hp_data,department)$n) %>% mutate(perc_hp=num_hp/2988) %>% mutate(perc_non=num_non_hp/(14999-2988))
dist_jobs %>% ggplot + geom_bar(mapping=aes(x=reorder(department,perc_hp),y=perc_hp),stat="identity",fill="lightcoral") + scale_y_continuous(breaks=seq(0,0.3,0.05)) + coord_flip() + xlab("Department") + ylab("Percentage of High Performers in Department")
dist_jobs %>% ggplot + geom_bar(mapping=aes(x=reorder(department,perc_non),y=perc_non),stat="identity",fill="steelblue") + scale_y_continuous(breaks=seq(0,0.3,0.05)) + coord_flip() + xlab("Department") + ylab("Percentage of Non High Performers in Department")
ggplot(hp_data) + geom_bar(mapping=aes(x=factor(salary,levels=c("low","medium","high")),y=(..count..)/sum(..count..)),fill="lightcoral") + xlab("Salary Grade") + ylab("Percentage")
ggplot(non_hp_data) + geom_bar(mapping=aes(x=factor(salary,levels=c("low","medium","high")),y=(..count..)/sum(..count..)),fill="steelblue") + xlab("Salary Grade") + ylab("Percentage")
ggplot(hp_data) + geom_bar(mapping=aes(x=number_project,y=(..count..)/sum(..count..)),fill="lightcoral") + scale_x_continuous(breaks=seq(2,7,1)) + xlab("Number of Projects") + ylab("Percentage")
corr_data = select(data,satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,high_performer)
corr_data = sapply( corr_data, as.numeric )
cor(corr_data,as.numeric(data$left))
cor.test(corr_data[,1],as.numeric(data$left))
corr_data_hp = select(hp_data,satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,high_performer)
corr_data_hp = sapply( corr_data_hp, as.numeric )
cor(corr_data_hp,as.numeric(hp_data$left))
cor.test(corr_data_hp[,4],as.numeric(hp_data$left))
left_data = filter(data,left==1)
ggplot(left_data) + geom_bar(mapping=aes(x=as_factor(salary,levels=c("low","medium","high")),y=(..count..)/sum(..count..)),fill="steelblue") + xlab("Salary Grade") + ylab("Percentage")
count(left_data,department) %>% ggplot() + geom_bar(mapping=aes(x=reorder(department,n),y=n),stat="identity",fill="steelblue") + coord_flip() + xlab("Department") + ylab("Count")
ggplot(top_leaving_depts) + geom_histogram(mapping=aes(x=satisfaction_level,y=..density..),fill="steelblue",binwidth = 0.1) + facet_wrap(~department) + xlab("Satisfaction") + ylab("Density")
ggplot(hp_data) + geom_point(mapping=aes(x=average_montly_hours,y=satisfaction_level,color=left)) + xlab("Monthly Average Hours") + ylab("Satisfaction Level")

#predictive modelling
library(rpart)
data= data[sample(1:nrow(data)),]
data= data[sample(1:nrow(data)),]
ddata= data[sample(1:nrow(data)),]
train_data=data[1:11999,]
test_data=data[11999:14999,]
y_train = select(train_data,left)
x_train = select(train_data,-left)
y_test = select(test_data,left)
x_test = select(test_data,-left)
x = train_data
fit = rpart(y_train$left~., data=x, method="class")
predicted= predict(fit,x_test)
predict_tibble = as_tibble(predicted)
predict_tibble = rename(predict_tibble,zero="0",one="1")
predict_tibble = mutate(predict_tibble,pred=ifelse(predict_tibble$zero>predict_tibble$one,0,1))
prediction = predict_tibble[,3]
mean(prediction == y_test)




