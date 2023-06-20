library("ggseg")
library("readr")
library("dplyr")
library("ggplot2")
library("svglite")

# Author: Tobias Kaufmann, Dominik Kraft

setwd("/path/to/correlation_tables")

hbn <- read.csv("hbn_feature_embedding_corr.csv", row.names = NULL)
d1 <- hbn %>% group_by(group) 

baseline <- read.csv("abcd_feature_embedding_corr_baseline.csv")
follow <- read.csv("abcd_feature_embedding_corr_follow.csv")
baseline$group=sprintf("baseline: %s",baseline$group)
follow$group=sprintf("follow-up: %s",follow$group)
abcd <- rbind(baseline, follow)
d2 <- abcd %>%   group_by(group)


d1$site="HBN: Correlations betweeen raw features and first embedding"
d2$site="ABCD: Correlations betweeen raw features and first embedding"

data=rbind(d1,d2)
data$group=factor(data$group, levels=unique(data$group) )
p=ggplot(data) +
  geom_brain(atlas = dk, 
             position = position_brain(. ~ side + hemi),
             aes(fill = r)) +
  facet_wrap(~group, ncol=1) + #ncols weg
  scale_fill_viridis_c(option = "magma", direction = 1) +
  theme_void() +
  theme(legend.position = "bottom", 
        legend.key.width = unit(1.5,"cm"),
        legend.key.height = unit(0.3,"cm"),
        strip.text = element_text(hjust = 0, face = "plain",size = 17),
        legend.title = element_text(face = "plain",size = 17),
        legend.text = element_text(face = "plain",size = 17)) 
p
ggsave(filename= "newfig.pdf",
       plot = p, width = 8, height = 10, bg="white")

