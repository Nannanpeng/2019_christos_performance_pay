library(plyr)
library(zoo)
library(mAr)
library(spam)
library(markovchain)
library(Hmisc)
library(Rcpp)
library(RcppCNPy)
library(dplyr)


#keep only data with all data or only savings missing (there are a lot of those)
mydata=read.csv("alldata.csv")
keep=(apply(is.na(mydata),1,sum)<=1)
mydata=mydata[keep,]
keep1=(apply(is.na(mydata),1,sum)==0)
mydata=mydata[keep1 | is.na(mydata$savings),]

#omit unemployed (there are only 153)
mydata=mydata[mydata$employed==1,]
mydata$employed=NULL
mydata$wage=mydata$labinc/mydata$hours
minimum_wage=min(mydata$wage)



#Trim super high wages
mydata$wage=mydata$labinc/mydata$hours
mydata=mydata[mydata$wage<=100,]

#define job numbers  numbers and lose occname and indname
occ=unique(mydata[,c("occname","indname")])
occ$job=1:nrow(occ)
mydata=join(mydata,occ,by=c("occname","indname"),type="left")
mydata$occname=NULL
mydata$indname=NULL

newocc=occ[order(occ$indname,occ$occname),]
newocc=newocc[!(newocc$indname==""),]
newocc$newjob=1:9

#index person numbers
persons=unique(mydata$person)
df=data.frame(person=persons,person_index=1:length(persons))
mydata=join(mydata,df,by="person",type="left")
mydata$person=mydata$person_index
mydata$person_index=NULL

#Find min and max ages for eache person
min_age=aggregate(mydata[,c("person","age")],by=list(person=mydata$person),min)
min_age=data.frame(person=min_age$person,min_age=min_age$age)
max_age=aggregate(mydata[,c("person","age")],by=list(person=mydata$person),max)
max_age=data.frame(person=max_age$person,max_age=max_age$age)
mydata=join(mydata,min_age,by="person")
mydata=join(mydata,max_age,by="person")
mydata$year=mydata$age-mydata$min_age-1


mydata$wage=mydata$labinc/mydata$hours
mydata$frac_savings=mydata$savings/mydata$labinc

mydata=mydata[,c("person","age","year","job","pp","wage","frac_savings","hours")]





#separate data with missing savings only
MS=mydata[is.na(mydata$frac_savings),]
mydata=mydata[!is.na(mydata$frac_savings),]
MS=MS[MS$person %in% unique(mydata$person),]



#Remove cases where savings is more than labour income
mydata=mydata[mydata$frac_savings<=1,]

initial_states=mydata[mydata$age==25 ,c('job','pp','wage')]
initial_states=join(initial_states,newocc,by="job")
initial_states$job=initial_states$newjob
initial_states=initial_states[,c('job','pp','wage')]
initial_states=initial_states[!is.na(initial_states$job),]
write.csv(initial_states,'initial_states.csv')

#get mean  of wage, frac_savings and hours by (age,job,pp)
mymean=aggregate(mydata[,c("age","job","pp","wage","frac_savings","hours")],by=list(mydata$age,mydata$job,mydata$pp),FUN=mean)
mymean=mymean[,!grepl("Group",colnames(mymean))]

#change mymean column names before joining with other data
old_cols=c("frac_savings","hours","wage")
new_cols=paste("mean_",old_cols,sep="")
replace=new_cols
names(replace)=old_cols
mymean=plyr::rename(mymean,replace=replace)

#get sd  of wage, frac_savings and hours by (age,job,pp)
mysd=aggregate(mydata[,c("age","job","pp","wage","frac_savings","hours")],by=list(myage=mydata$age,myjob=mydata$job,mypp=mydata$pp),sd)
mysd=mysd[,c("myage","myjob","mypp","wage","frac_savings","hours")]

#change mysd column names before joining with other data
old_cols=c("frac_savings","hours","wage")
new_cols=paste("sd_",old_cols,sep="")
replace=new_cols
names(replace)=old_cols
mysd=plyr::rename(mysd,replace=replace)
mysd=plyr::rename(mysd,c("myage"="age","myjob"="job","mypp"="pp"))


mylogapprox<-function(x,y,xout){
  y=log(y)
  inside=(xout>=min(x) & xout<=max(x))
  outside=!inside
  yout=rep(NA,length(xout))
  yout[inside]=approx(x,y,xout=xout[inside])$y
  df=data.frame(x=x,y=y)
  model=lm(y ~ x,data=df)
  yout[outside]=predict(model,data.frame(x=xout[outside]))
  yout=exp(yout)
  y=exp(y)
  yout[xout %in% x]=y
  return(yout)
  
}
myapprox<-function(x,y,xout){
  inside=(xout>=min(x) & xout<=max(x))
  outside=!inside
  yout=rep(NA,length(xout))
  yout[inside]=approx(x,y,xout=xout[inside])$y
  yout[outside]=rnorm(sum(outside),mean(y),sd(y))
  return(yout)
  
}

#interpolate and extrapolate mean and sd for ages we don't have
df_mean=data.frame()
df_sd=data.frame()
for (job in unique(mymean$job)){
  for (pp in unique(mymean$pp)){
    print(paste(job,pp))
    x=( (mymean$job==job) & (mymean$pp==pp) )
    
    #Fill in means
    new_data=data.frame(mean_wage=mylogapprox(mymean$age[x],mymean$mean_wage[x],25:55))
    new_data$job=job
    new_data$pp=pp
    new_data$age=25:55
    new_data$mean_frac_savings=myapprox(mymean$age[x],mymean$mean_frac_savings[x],25:55)
    new_data$mean_hours=myapprox(mymean$age[x],mymean$mean_hours[x],25:55)
    df_mean=rbind(df_mean,new_data)
    
    #Fill in sds
    x=( (mysd$job==job) & (mysd$pp==pp) )
    new_data=data.frame(sd_wage=mylogapprox(mysd$age[x],mysd$sd_wage[x],25:55))
    new_data$job=job
    new_data$pp=pp
    new_data$age=25:55
    new_data$sd_frac_savings=myapprox(mysd$age[x],mysd$sd_frac_savings[x],25:55)
    new_data$sd_hours=myapprox(mysd$age[x],mysd$sd_hours[x],25:55)
    df_sd=rbind(df_sd,new_data)
    
    
    
  }
}
mymean=df_mean
mysd=df_sd




# where SD is NAN because of only one entry set SD=0
x=(apply(is.na(mysd),1,sum)>0)
mysd[x,c("sd_wage","sd_frac_savings","sd_hours")]=0




#add mean and sd data to mydata
mydata=join(mydata,mymean,by=c("age","job","pp"))
mydata=join(mydata,mysd,by=c("age","job","pp"))
MS=join(MS,mymean,by=c("age","job","pp"))
MS=join(MS,mysd,by=c("age","job","pp"))
MS=MS[apply(is.na(MS),1,sum)<=1,]



#normalize wage,frac_savings and hours 
mydata$zs=mydata$frac_savings/mydata$mean_frac_savings
mydata$zw=(mydata$wage-mydata$mean_wage)/mydata$sd_wage
mydata$zh=(mydata$hours-mydata$mean_hours)/mydata$sd_hours
MS$zw=(MS$wage-MS$mean_wage)/MS$sd_wage
MS$zh=(MS$hours-MS$mean_hours)/MS$sd_hours
MS$frac_savings=NULL
MS=MS[apply(is.na(MS),1,sum)==0,]


#Remove any rows with NAs
keep=(apply(is.na(mydata),1,sum)==0)
mydata=mydata[keep,] 

#***************************************************************************
#interpolate data for missing ages linearly
#extrapolate back to age 25 with multivariate AR1 on normalized (zs,zh,zw)
#extrapolate forward to age 55 but use data where we have it from MS
#which is the post age 43 data with just savings missing
DF=data.frame()
n=length(unique(mydata$person))
persons=unique(mydata$person)
for (i in 1:length(persons)){
  person=persons[i]
  
  
  X=mydata[mydata$person==person,c("age","job","pp","zs","zh","zw")]
  #Y=MS[MS$person==person,c("age","job","pp","zh","zw")]
  min_age=min(X$age)
  max_age=max(X$age)
  df=data.frame(age=seq(min_age,max_age,1))
  
  df$job=NA
  df[X$age-min_age+1,"job"]=X$job
  df=df[order(df$age),]
  df$job=na.locf(df$job)
  
  df$pp=NA
  df[X$age-min_age+1,"pp"]=X$pp
  df$pp=na.locf(df$pp)
  
  
  
  if (nrow(X)<max_age-min_age+1){
    #if (nrow(X)>5){
    #degf=min(10,nrow(X)-1)
    # df$zs=predict(smooth.spline(X$age,X$zs,df=degf),df$age)$y
    #df$zw=predict(smooth.spline(X$age,X$zs,df=degf),df$age)$y
    #df$zh=predict(smooth.spline(X$age,X$zs,df=degf),df$age)$y
    #}else{
    df$zs=approx(X$age,X$zs,df$age)$y
    df$zw=approx(X$age,X$zw,df$age)$y
    df$zh=approx(X$age,X$zh,df$age)$y
    
    # }
    
  }else{
    df=X[order(X$age),]
  }
  df$person=person
  df$year=df$age-min_age+1
  
  
  DF=rbind(DF,df)
  print(paste(i,n))
  
}

#************************************************************************************
# save(DF,file="normalized_trajectories")
# 
# load("normalized_trajectories")
# #Get discrete points for zw,zs,zh
# DF1=DF
# state_to_shw=list()
# for (var in c("zw","zh","zs")){
#   z=DF[,var]
#   zstates=1+floor(49*(z-min(z))/(max(z)-min(z)))
#   state_var=paste(var,"_state",sep="")
#   df1=data.frame(z=z,zstate=zstates)
#   state_to_shw[[var]]=aggregate(df1,by=list(df1$zstate),mean)
#   state_to_shw[[var]]=state_to_shw[[var]][,c("zstate","z")]
# }
# 
# 
# 
# #Fit Markov chain to (zw,zs,zh) state transitions
# get_state_sequence<-function(person,var){
#   X=DF1[DF1$person==person,c("year",var)]
#   X=X[order(X$year),]
#   return(X[,var])
# }
# shw_MC=list()
# shw_steady=list()
# for (var in c("zw","zh","zs")){
#   state_var=paste(var,"_state",sep="")
#   sequences=sapply(unique(DF1$person),get_state_sequence,state_var)
#   shw_MC[[var]]=markovchainFit(sequences)$estimate
#   shw_steady[[var]]=steadyStates(shw_MC[[var]])
# }
# 
# 
# 
# #Fit Markov Chain  to job x pp
# get_state_sequence<-function(person){
#   X=DF1[DF1$person==person,c("year","job","pp")]
#   X=X[order(X$year),]
#   return(paste(X$job,X$pp,sep=":"))
# }
# sequences=sapply(unique(D1F$person),get_state_sequence)
# jobpp_MC=markovchainFit(sequences)$estimate
# jobpp_steady=steadyStates(jobpp_MC)
# #************************************************************************************
# #Run simulations 
# vars=c("zw","zs","zh")
# state_vars=paste(vars,"_state",sep="")
# 
# 
# get_simulation<-function(min_age,max_age,jobpp,shw_MC,jobpp_steady,shw_steady){
#   num_years=max_age-min_age+1
#   new_jobpp=markovchainSequence(num_years,jobpp_MC,t0=sample(jobpp_MC@states,size=1,prob=jobpp_steady))
#   job=as.numeric(as.vector(sapply(new_jobpp,function(x) strsplit(x,split=":")[[1]][[1]] )))
#   pp=as.numeric(as.vector(sapply(new_jobpp,function(x) strsplit(x,split=":")[[1]][[2]] )))
#   age=min_age:max_age
#   shw=list()
#   zshw=list()
#   for (var in vars){
#     shw[[var]]=as.numeric(markovchainSequence(num_years,shw_MC[[var]],t0=sample(shw_MC[[var]]@states,size=1,prob=shw_steady[[var]])))
#     df=data.frame(zstate=shw[[var]])
#     zshw[[var]]=join(df,state_to_shw[[var]],by="zstate")
#   }
#   zs=zshw[["zs"]]$z
#   zh=zshw[["zh"]]$z
#   zw=zshw[["zw"]]$z
#   X=data.frame(age=age,job=job,pp=pp,zs=zs,zw=zw,zh=zh)
#   return(X)
# }
# 
# 
# DF2=data.frame()
# for (i in 1:(24*1000)){
#   X=get_simulation(min_age,max_age,jobpp,shw_MC,jobpp_steady,shw_steady)
#   X$person=i
#   DF2=rbind(DF2,X)
#   if ( (i %% 1000)==0 ){print(i)}
# }
# 
# save(DF2,file="simulated_data.RData")
# 
# mycounts=aggregate(DF2,by=list(DF2$age,DF2$job,DF2$pp),FUN=function(x) length(x))
#***********************************************************************************
load("simulated_data.RData")
#convert back to absolute wage,hours and savings, so we can calculate assetts
DF3=DF2
DF3=join(DF3,mymean,by=c("age","job","pp"))
DF3=join(DF3,mysd,by=c("age","job","pp"))
DF3$wage=DF3$mean_wage+DF3$zw*DF3$sd_wage  
DF3$hours=DF3$mean_hours+DF3$zh*DF3$sd_hours
DF3$frac_savings=DF3$mean_frac_savings*DF3$zs

DF3$frac_savings[DF3$frac_savings>1]=1
DF3$frac_savings[DF3$frac_savings<0]=0
DF3$wage[DF3$wage<0]=0
DF3$hours[DF3$hours<0]=0
DF3$savings=DF3$frac_savings*DF3$wage*DF3$hours



#Calculate assetts
cppFunction('NumericMatrix get_assets(NumericMatrix savings,double rate) {
            int nrow = savings.nrow(), ncol = savings.ncol();
            int i; int j;
            NumericMatrix assets(nrow,ncol);
            for (j=0; j<ncol; j++){
            assets(0,j)=savings(0,j);
            for (i=1; i<nrow ; i++){
            assets(i,j)=(1+rate)*assets(i-1,j)+savings(i,j);
            }
            }
            return assets;
            }')
savings=matrix(DF3$savings,nrow=55-25+1)
rate=.04
assets=22000+get_assets(savings,rate)
assets=matrix(assets,ncol=1)
DF3$assets=assets

DF4=DF3[,c("age","job","pp","assets","wage","savings","hours")]
DF4=DF4[!is.na(DF4$assets),]
mean_assets=aggregate(DF4,by=list(myage=DF4$age,myjob=DF4$job,mypp=DF4$pp),FUN=mean)
mean_assets=mean_assets[,c("myage","myjob","mypp","assets")]
colnames(mean_assets)=c("age","job","pp","mean_assets")

#This will change job numbers to coincide to the way Christos defined them
DF6=join(DF4,newocc,by="job")
DF6$old_job=DF6$job
DF6$job=DF6$newjob
DF6$newjob=NULL
DF6=DF6[!is.na(DF6$job),]

#initial population for forward simulation
DF6=DF6[DF6$age==25,c("job","pp","wage")]
initial_mean=aggregate(DF6,by=list(myjob=DF6$job,mypp=DF6$pp),FUN=mean)
initial_mean$mean_wage=initial_mean$wage
initial_mean$wage=NULL
initial_sd=aggregate(DF6,by=list(myjob=DF6$job,mypp=DF6$pp),FUN=sd)
initial_sd$sd_wage=initial_sd$wage
initial_sd$wage=NULL
initial_mean$job=NULL
initial_mean$pp=NULL
initial_sd$job=NULL
initial_sd$pp=NULL
colnames(initial_mean)=c("job","pp","mean_wage")
colnames(initial_sd)=c("job","pp","sd_wage")
DF7=DF6[,c("job","pp")]
DF7=join(DF7,initial_mean,by=c("job","pp"))
DF7=join(DF7,initial_sd,by=c("job","pp"))

num_initial_agents=100000
DF8=data.frame()
while (nrow(DF8)<num_initial_agents){
  df=DF7[,c("job","pp")]
  df$wage=rnorm(nrow(df),mean=DF7$mean_wage,sd=DF7$sd_wage)
  DF8=rbind(DF8,df)
  
}
DF8=sample_n(DF8,size=num_initial_agents)
DF8$wage[DF8$wage<minimum_wage]=minimum_wage
npySave(filename="initial_states.npy",as.matrix(DF8))





sd_assets=aggregate(DF4,by=list(myage=DF4$age,myjob=DF4$job,mypp=DF4$pp),FUN=sd)
sd_assets=sd_assets[,c("myage","myjob","mypp","assets")]
colnames(sd_assets)=c("age","job","pp","sd_assets")

mymean1=join(mymean,mean_assets,by=c("age","job","pp"))
mysd1=join(mysd,sd_assets,by=c("age","job","pp"))
mymean=mymean1
mysd=mysd1

meansd=join(mymean,mysd,by=c("age","job","pp"))
meansd=meansd[,c("age","job","pp","mean_assets","mean_wage","mean_frac_savings",
                 "mean_hours",
                 "sd_assets","sd_wage","sd_frac_savings",
                 "sd_hours")]
meansd=meansd[order(meansd$age,meansd$job,meansd$pp),]
mydata=meansd
mydata$age=NULL
mydata$job=NULL
mydata$pp=NULL
M=as.matrix(mydata)

save(mymean,mysd,file="mean_sd.RData")

num_bins=9
exp_deciles=c(qexp(seq(0,1-1/num_bins,1/num_bins)),qexp(.99))
max_decile_exp=max(exp_deciles)
exp_breaks=round(2**16*exp_deciles/max_decile_exp)
exp_breaks[1]=1
lookup_table=rep(0,2**16)
for (i in 1:num_bins){
  lower_limit=exp_breaks[i]
  upper_limit=exp_breaks[i+1]
  lookup_table[lower_limit:upper_limit]=i+seq(0,1,1/(upper_limit-lower_limit))
}
exp_lookup_table=lookup_table
exp_midpoints=qexp(seq(.5*1/(num_bins),1,1/(num_bins)))
exp_offset=0
exp_multiplier=max_decile_exp/2**16


num_bins=9
norm_deciles=c(qnorm(.00001),qnorm(seq(1/num_bins,1-1/num_bins,1/num_bins)),qnorm(.99))
min_decile_norm=min(norm_deciles)
max_decile_norm=max(norm_deciles)
norm_width=max_decile_norm-min_decile_norm
norm_breaks=round(2**16*(norm_deciles-min_decile_norm)/norm_width)
lookup_table=rep(0,2**16)
for (i in 1:num_bins){
  lower_limit=max(1,norm_breaks[i])
  upper_limit=norm_breaks[i+1]
  lookup_table[lower_limit:upper_limit]=i+seq(0,1,1/(upper_limit-lower_limit))
}
norm_lookup_table=lookup_table
norm_lower_bound=min_decile_norm
norm_midpoints=qnorm(seq(.5*1/(num_bins),1,1/(num_bins)))
norm_offset=min_decile_norm
norm_multiplier=(max_decile_norm-min_decile_norm)/2**16


DF5=DF4
DF5$frac_savings=DF5$savings/(DF5$wage*DF5$hours)
DF5$savings=NULL
DF5=DF5[!is.nan(DF5$frac_savings),]
overall_mean=apply(DF5[,c("assets","wage","frac_savings","hours")],2,mean)
overall_sd=apply(DF5[,c("assets","wage","frac_savings","hours")],2,sd)
overall_min=apply(DF5[,c("assets","wage","frac_savings","hours")],2,min)
dist_type=c(0,1,0,1)

dist_data=rbind(overall_mean,overall_sd,dist_type)
colnames=c("assets","wage","frac_savings","hours")
index=paste("index_",colnames,sep="")

df=data.frame(row=1:num_bins)
df1=data.frame(row=1:num_bins)
df_index=data.frame(row=1:num_bins)
df_breaks=data.frame(row=1:(num_bins+1))
offset=list()
multiplier=list()
mymax=list()
for (i in 1:4){
    print(i)
  type=dist_type[i]
  if (type==0){
    col=round(exp_midpoints/exp_multiplier)
    offset[[i]]=0
    multiplier[[i]]=exp_multiplier*overall_mean[i]
    #col=exp_midpoints
    col1=exp_midpoints*overall_mean[i]
    colbreaks=round(2**16*exp_deciles/max_decile_exp)
    mymax[[i]]=overall_mean[i]*max_decile_exp
  }else{
    col=round( (norm_midpoints-norm_offset)/norm_multiplier )
    offset[[i]]=overall_mean[i]+norm_offset*overall_sd[i]
    multiplier[[i]]=norm_multiplier*overall_sd[i]
    #col=norm_midpoints
    col1=overall_mean[i]+norm_midpoints*overall_sd[i]
    colbreaks=round(2**16*(overall_mean[i]+norm_deciles*overall_sd[i])/(overall_mean[i]+max_decile_norm*overall_sd[i]))
    mymax[[i]]=overall_mean[i]+max_decile_norm*overall_sd[i]
  }
  col[col<0]=min(col[col>0])/2
  col1[col1<0]=min(col1[col1>0])/2
  colbreaks[colbreaks<0]=min(colbreaks[colbreaks>0])/2
  mycol=data.frame(col)
  mycol1=data.frame(col1)
  mycolbreaks=data.frame(colbreaks)
  colnames(mycol)=colnames[i]
  colnames(mycol1)=colnames[i]
  colnames(mycolbreaks)=colnames[i]
  icol=data.frame(1:num_bins)
  colnames(icol)=index[i]
  df=cbind(df,mycol)
  df1=cbind(df1,mycol1)
  df_index=cbind(df_index,icol)
  df_breaks=cbind(df_breaks,mycolbreaks)
  
}
df=df[,2:5]
df_index=df_index[,2:5]
df_breaks=df_breaks[2:ncol(df_breaks)]
df_mid=df
offset=unlist(offset)
multiplier=unlist(multiplier)

offset_multiplier=data.frame(offset=offset,multiplier=multiplier)



for (i in 1:ncol(df_mid)){
  df_mid[,i]=round(2**16*df_mid[,i]/mymax[[i]])
}
scale=unlist(mymax)/2**16


lookup_table=data.frame()
for (j in 1:ncol(df_breaks)){
  row=rep(0,2**16)
  for (i in 2:nrow(df_breaks)){
    lower_limit=df_breaks[i-1,j]
    upper_limit=df_breaks[i,j]
    row[lower_limit:upper_limit]=i-1
  }
  lookup_table=rbind(lookup_table,row)
  
}

#*******************************************************************
num_bins=10
exp_deciles=qexp(c(seq(1/num_bins,1-1/num_bins,1/num_bins),.99))
norm_deciles=qnorm(c(seq(1/num_bins,1-1/num_bins,1/num_bins),.99))
exp_lookup_table=pexp(seq(0,max(exp_deciles)*(1-1/2**16),max(exp_deciles)/2**16))
norm_lookup_table=pnorm(seq(min(norm_deciles),max(norm_deciles)-(max(norm_deciles)-min(norm_deciles))/2**16,(max(norm_deciles)-min(norm_deciles))/2**16))

exp_offset=0
exp_multiplier=max(exp_deciles)/2**16
norm_offset=min(norm_deciles)
norm_multiplier=(max(norm_deciles)-min(norm_deciles))/2**16

df=data.frame()
df1=data.frame()
for (i in 1:4){
  type=dist_type[i]
  if (type==0){
    col=round(exp_deciles/exp_multiplier)
    col1=round(exp_deciles/exp_multiplier)
  }else{
    col=round( (norm_deciles-norm_offset)/norm_multiplier )
  }
  df=rbind(df,col)
}
df=t(df)
rownames(df)=NULL
colnames(df)=colnames

exp_multipliers=exp_multiplier*overall_mean[c(1,3)]
norm_offsets=overall_mean[c(2,4)]+norm_offset*overall_sd[c(2,4)]
norm_multipliers=norm_multiplier*overall_sd[c(2,4)]
offsets=rep(0,4)
multipliers=rep(0,4)
offsets[c(2,4)]=norm_offsets
multipliers[c(1,3)]=exp_multipliers
multipliers[c(2,4)]=norm_multipliers
names(offsets)=colnames
names(multipliers)=colnames
offset_multipliers=data.frame(offset=offsets,multiplier=multipliers)
df1=data.frame(offset=rep(norm_offset,2),multiplier=rep(norm_multiplier,2),row.names=c('Z','newZ'))
offset_multipliers=rbind(offset_multipliers,df1)
offset_multipliers$name=rownames(offset_multipliers)
offset_multipliers=offset_multipliers[,c('name','offset','multiplier')]
write.csv(offset_multipliers,file="offset_multipliers")



DF=expand.grid(as.data.frame(df[,ncol(df):1]))
DF=DF[,c("assets","wage","frac_savings","hours")]
grid=unlist(t(DF))
flat_grid_data_1=rep(grid,31*9*2*2)

col=round( (norm_deciles-norm_offset)/norm_multiplier )
flat_grid_data_2=col
flat_grid_data_3=col

flat_grid_data=c(flat_grid_data_1,flat_grid_data_2,flat_grid_data_3)

flat_grid_data[flat_grid_data==65536]=65535
npySave("flat_grid_data.npy",flat_grid_data)
npySave("norm_lookup_table.npy",norm_lookup_table)
npySave("exp_lookup_table.npy",exp_lookup_table)


structure=read.csv('data/structure.csv',stringsAsFactors=FALSE)
rownames(structure)=structure$name
structure[rownames(offset_multipliers),c('name','offset','multiplier')]=offset_multipliers
write.csv(structure,file='./data/structure.csv',row.names=FALSE)
#********************************************************************************


