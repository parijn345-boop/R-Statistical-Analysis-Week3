# Week 3: Statistical Analysis and Predictive Modeling using R
# Built-in iris dataset; base R only.
data("iris"); df <- iris
cat("Dimensions:",nrow(df),"rows x",ncol(df),"columns\n")
cat("Missing cells:",sum(is.na(df)),"\n"); str(df); summary(df)
num <- df[,c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width")]
cat("\nCorrelation matrix:\n"); print(round(cor(num),3))
print(cor.test(df$Sepal.Length,df$Petal.Length))
anova_fit <- aov(Petal.Length ~ Species,data=df); print(summary(anova_fit)); print(TukeyHSD(anova_fit)); print(shapiro.test(residuals(anova_fit)))
model <- lm(Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Species,data=df)
cat("\nRegression summary:\n"); print(summary(model)); print(shapiro.test(residuals(model)))
set.seed(42); train_idx <- sample(seq_len(nrow(df)),floor(.8*nrow(df))); train <- df[train_idx,]; test <- df[-train_idx,]
hold_model <- lm(Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Species,data=train); hold_pred <- predict(hold_model,newdata=test)
rmse <- sqrt(mean((test$Petal.Length-hold_pred)^2)); mae <- mean(abs(test$Petal.Length-hold_pred)); r2 <- 1-sum((test$Petal.Length-hold_pred)^2)/sum((test$Petal.Length-mean(test$Petal.Length))^2)
cat("Holdout RMSE:",rmse," MAE:",mae," R2:",r2,"\n")
set.seed(42); fold_id <- sample(rep(1:5,length.out=nrow(df))); cv <- data.frame(Fold=1:5,RMSE=NA_real_,MAE=NA_real_,R2=NA_real_)
for(k in 1:5){ tr<-df[fold_id!=k,]; te<-df[fold_id==k,]; fit<-lm(Petal.Length ~ Sepal.Length+Sepal.Width+Petal.Width+Species,data=tr); pr<-predict(fit,newdata=te); cv$RMSE[k]<-sqrt(mean((te$Petal.Length-pr)^2)); cv$MAE[k]<-mean(abs(te$Petal.Length-pr)); cv$R2[k]<-1-sum((te$Petal.Length-pr)^2)/sum((te$Petal.Length-mean(te$Petal.Length))^2) }
print(cv); print(colMeans(cv[,c("RMSE","MAE","R2")]))
png("week3_species_boxplot.png",1000,650,res=130); boxplot(Petal.Length~Species,data=df,col="lightblue",main="Petal Length by Species",xlab="Species",ylab="Petal length (cm)"); dev.off()
png("week3_correlation_matrix.png",1000,800,res=130); image(1:4,1:4,t(cor(num)[4:1,]),axes=FALSE,main="Correlation Matrix"); axis(1,1:4,labels=colnames(num),las=2); axis(2,1:4,labels=rev(colnames(num)),las=2); dev.off()
png("week3_observed_fitted.png",1000,650,res=130); plot(df$Petal.Length,fitted(model),pch=19,xlab="Observed petal length",ylab="Fitted petal length",main="Observed vs Fitted"); abline(0,1,lty=2); dev.off()
png("week3_residuals_fitted.png",1000,650,res=130); plot(fitted(model),residuals(model),pch=19,xlab="Fitted values",ylab="Residuals",main="Residuals vs Fitted"); abline(h=0,lty=2); dev.off()
png("week3_qq_plot.png",1000,650,res=130); qqnorm(residuals(model),main="Normal Q-Q Plot"); qqline(residuals(model),col="red"); dev.off()
write.csv(cv,"week3_cv_metrics.csv",row.names=FALSE); cat("Saved diagnostic plots and cross-validation CSV.\n")
