 ################################################################################
 ########################## CARICAMENTO LIBRERIE    #############################
 ################################################################################

## ----message=FALSE, warning=FALSE, include=FALSE------------------------------

suppressPackageStartupMessages({
library(conflicted)
library(BiocManager)
library(org.Hs.eg.db)
library(TCGAbiolinks)
library(SummarizedExperiment)
library(dplyr)
library(DT)
library("writexl")
library("DESeq2")
library("iSEEde")
library("iSEE")
library("airway")
library("apeglm")
library(survminer)
library(survival)
library(forestmodel)
library("pheatmap")
library(tidyverse)
library(biomaRt)
})

#  ################################################################################
#  ################################### LOAD #######################################
#  ################################################################################

setwd("/Users/gianmariazaccaria/Documents/R_studio/PAAD/TCGA-PAAD/RNA-seq")
load(gsub(" ","",paste(getwd(),"/whole_RNAseq.rda")));

#Gene annotation
rownames(data) <- data@rowRanges$gene_name

###############################################################################
########################### ANALISI DEG con DESeq2 ############################
###############################################################################

#KRAS   ########################################################################
#DeSeq2
table(data$KRAS)
data$KRAS <- relevel(as.factor(data$KRAS), "0"); table(data$KRAS)

dds <- DESeqDataSet(data, ~ KRAS)
dds$condition <- factor(dds$KRAS, levels = c("0","1"))
dds <- DESeq(dds); dds
res_KRAS <- results(dds, alpha = 0.05); summary(res_KRAS)

#Ordering pval adj
resOrdered <- res_KRAS[order(res_KRAS$padj),]; resOrdered

##Conto i geni con p adjusted <0.05 e con LFC > 1
sum(abs(resOrdered$log2FoldChange)>1 & resOrdered$padj < 0.05, na.rm=TRUE)

#Filtering and ordering DEGs
resFiltered <- resOrdered[ which(resOrdered$padj < 0.05 & abs(resOrdered$log2FoldChange)>1), ]; resFiltered
# resFil_Ord <- resFiltered[order(resFiltered$padj),]; resFil_Ord
# write.csv(resFiltered, file="./output/DE_results_filtered_KRAS.csv")

# #Extracting genes
# ensemble <- resFiltered@rownames; ensemble

#Filtro solo i DEGs dei geni
dds_filt <- dds[ensemble, ]

#Ordering according to the outcome
dds_filt_ord <- dds_filt[,order(dds_filt$condition)]

resultsNames(dds_filt)

#Information of genes
DEG_res <- as.data.frame(dds_filt_ord@assays@data@listData[["counts"]])
DEG_res$genes <- dds_filt_ord@rowRanges$ensemble

#Data extraction
write.csv(DEG_res,"./output/DEG_KRAS.csv", dec = ".", sep = " ", eol = "\n",
          row.names = TRUE, col.names = TRUE)

#SMAD4   ########################################################################
#DeSeq2
table(data$SMAD4)
data$SMAD4 <- relevel(as.factor(data$SMAD4), "0"); table(data$SMAD4)

dds <- DESeqDataSet(data, ~ SMAD4)
dds$condition <- factor(dds$SMAD4, levels = c("0","1"))
dds <- DESeq(dds)

res_SMAD4 <- results(dds, alpha = 0.05); summary(res_SMAD4)

#Ordering pval adj
resOrdered <- res_SMAD4[order(res_SMAD4$padj),]; resOrdered

##Conto i geni con p adjusted <0.05 e con LFC > 1
sum(abs(resOrdered$log2FoldChange) > 1 & resOrdered$padj < 0.05, na.rm=TRUE)

#Filtering and ordering DEGs
resFiltered <- resOrdered[which(resOrdered$padj < 0.05 & abs(resOrdered$log2FoldChange) > 1), ]; resFiltered
# resFil_Ord <- resFiltered[order(resFiltered$padj),]; resFil_Ord
# write.csv(resFiltered, file="./output/DE_results_filtered_SMAD4.csv")

#Extracting genes
ensemble <- resFiltered@rownames; ensemble
# write.csv(data_frame(transcripts),"/Users/gianmariazaccaria/Documents/R_studio/CPTAC-PDA/genomic/RNAseq/Results/transcripts.csv", row.names=FALSE)

#Filtro solo i DEGs dei geni
dds_filt <- dds[ensemble, ]

#Ordering according to the outcome
dds_filt_ord <- dds_filt[,order(dds_filt$condition)]

resultsNames(dds_filt)

#Information of genes
DEG_res <- as.data.frame(dds_filt_ord@assays@data@listData[["counts"]])
DEG_res$genes <- dds_filt_ord@rowRanges$ensemble

#Data extraction
write.csv(DEG_res,"./output/DEG_SMAD4.csv", dec = ".", sep = " ", eol = "\n",
          row.names = TRUE, col.names = TRUE)

#TP53   ########################################################################
#DeSeq2
table(data$TP53)
data$TP53 <- relevel(as.factor(data$TP53), "0"); table(data$TP53)

dds <- DESeqDataSet(data, ~ TP53)
dds$condition <- factor(dds$TP53, levels = c("0","1"))
dds <- DESeq(dds)

res_TP53 <- results(dds, alpha = 0.05);

#Ordering pval adj
resOrdered <- res_TP53[order(res_TP53$padj),]; resOrdered

##Conto i geni con p adjusted <0.05 e con LFC > 1
sum(abs(resOrdered$log2FoldChange) > 1 & resOrdered$padj < 0.05, na.rm=TRUE)

#Filtering and ordering DEGs
resFiltered <- resOrdered[ which(resOrdered$padj < 0.05 & abs(resOrdered$log2FoldChange) > 1), ]; resFiltered
# resFil_Ord <- resFiltered[order(resFiltered$padj),]; resFil_Ord
write.csv(resFiltered, file="./output/DE_results_filtered_TP53.csv")

#Extracting genes
ensemble <- resFiltered@rownames; ensemble
# write.csv(data_frame(transcripts),"/Users/gianmariazaccaria/Documents/R_studio/CPTAC-PDA/genomic/RNAseq/Results/transcripts.csv", row.names=FALSE)

#Filtro solo i DEGs dei geni
dds_filt <- dds[ensemble, ]

#Ordering according to the outcome
dds_filt_ord <- dds_filt[,order(dds_filt$condition)]

resultsNames(dds_filt)

#Information of genes
DEG_res <- as.data.frame(dds_filt_ord@assays@data@listData[["counts"]])
DEG_res$genes <- dds_filt_ord@rowRanges$ensemble

#Data extraction
write.csv(DEG_res,"./output/DEG_TP53.csv", dec = ".", sep = " ", eol = "\n",
          row.names = TRUE, col.names = TRUE)

#CDKN2A   ########################################################################
#DeSeq2
table(data$CDKN2A)
data$CDKN2A <- relevel(as.factor(data$CDKN2A), "0"); table(data$CDKN2A)

dds <- DESeqDataSet(data, ~ CDKN2A)
dds$condition <- factor(dds$CDKN2A, levels = c("0","1"))
dds <- DESeq(dds)

res_CDKN2A <- results(dds, alpha = 0.05);

#Ordering pval adj
resOrdered <- res_CDKN2A[order(res_CDKN2A$padj),]; resOrdered

##Conto i geni con p adjusted <0.05 e con LFC > 1
sum(abs(resOrdered$log2FoldChange) > 1 & resOrdered$padj < 0.05, na.rm=TRUE)

#Filtering and ordering DEGs
resFiltered <- resOrdered[ which(resOrdered$padj < 0.05 & abs(resOrdered$log2FoldChange) > 1), ]; resFiltered
# resFil_Ord <- resFiltered[order(resFiltered$padj),]; resFil_Ord
write.csv(resFiltered, file="./output/DE_results_filtered_CDKN2A.csv")

#Extracting genes
ensemble <- resFiltered@rownames; ensemble
# write.csv(data_frame(transcripts),"/Users/gianmariazaccaria/Documents/R_studio/CPTAC-PDA/genomic/RNAseq/Results/transcripts.csv", row.names=FALSE)

#Filtro solo i DEGs dei geni
dds_filt <- dds[ensemble, ]

#Ordering according to the outcome
dds_filt_ord <- dds_filt[,order(dds_filt$condition)]

resultsNames(dds_filt)

#Information of genes
DEG_res <- as.data.frame(dds_filt_ord@assays@data@listData[["counts"]])
DEG_res$genes <- dds_filt_ord@rowRanges$ensemble
#
# #Data extraction
# write.csv(DEG_res,"./output/DEG_CDKN2A.csv", dec = ".", sep = " ", eol = "\n",
#           row.names = TRUE, col.names = TRUE)

################################################################################
############################## DATA VISUALIZATION  #############################
################################################################################
detach("package:iSEEde", unload = TRUE)
detach("package:iSEE", unload = TRUE)
library(EnhancedVolcano)

p1 <- EnhancedVolcano(res_KRAS,
                lab = rownames(res_KRAS),
                x = 'log2FoldChange',
                y = 'pvalue',
                title = "KRAS",
                pCutoff = 10e-5,
                subtitle = NULL,
                caption = NULL,
                col = c('grey30', 'forestgreen', 'royalblue', 'red2'),
                pointSize = 4.5,
                labSize = 4.5,
                # shapeCustom = keyvals.shape,
                colCustom = NULL,
                colAlpha = 1,
                # legendLabSize = 15,
                legendPosition = 'none',
                # legendIconSize = 5.0,
                drawConnectors = TRUE,
                widthConnectors = 0.5,
                colConnectors = 'grey50',
                gridlines.major = TRUE,
                gridlines.minor = FALSE,
                border = 'partial',
                borderWidth = 1.5,
                borderColour = 'black')
p1

p2 <- EnhancedVolcano(res_SMAD4,
                      lab = rownames(res_SMAD4),
                      x = 'log2FoldChange',
                      y = 'pvalue',
                      title = "SMAD4",
                      pCutoff = 10e-5,
                      subtitle = NULL,
                      caption = NULL,
                      col = c('grey30', 'forestgreen', 'royalblue', 'red2'),
                      pointSize = 4.5,
                      labSize = 4.5,
                      # shapeCustom = keyvals.shape,
                      colCustom = NULL,
                      colAlpha = 1,
                      # legendLabSize = 15,
                      legendPosition = 'none',
                      # legendIconSize = 5.0,
                      drawConnectors = TRUE,
                      widthConnectors = 0.5,
                      colConnectors = 'grey50',
                      gridlines.major = TRUE,
                      gridlines.minor = FALSE,
                      border = 'partial',
                      borderWidth = 1.5,
                      borderColour = 'black')
# p2

p3 <- EnhancedVolcano(res_TP53,
                      lab = rownames(res_TP53),
                      x = 'log2FoldChange',
                      y = 'pvalue',
                      title = "TP53",
                      pCutoff = 10e-5,
                      subtitle = NULL,
                      caption = NULL,
                      col = c('grey30', 'forestgreen', 'royalblue', 'red2'),
                      pointSize = 4.5,
                      labSize = 4.5,
                      # shapeCustom = keyvals.shape,
                      colCustom = NULL,
                      colAlpha = 1,
                      # legendLabSize = 15,
                      legendPosition = "none",
                      # legendIconSize = NULL,
                      drawConnectors = TRUE,
                      widthConnectors = 0.5,
                      colConnectors = 'grey50',
                      gridlines.major = TRUE,
                      gridlines.minor = FALSE,
                      border = 'partial',
                      borderWidth = 1.5,
                      borderColour = 'black')
# p3

p4 <- EnhancedVolcano(res_CDKN2A,
                      lab = rownames(res_CDKN2A),
                      x = 'log2FoldChange',
                      y = 'pvalue',
                      title = "CDKN2A",
                      pCutoff = 10e-5,
                      subtitle = NULL,
                      caption = NULL,
                      pointSize = 4.5,
                      labSize = 4.5,
                      # shapeCustom = keyvals.shape,
                      colCustom = NULL,
                      colAlpha = 1,
                      # legendLabSize = 15,
                      legendPosition = "none",
                      # legendIconSize = NULL,
                      drawConnectors = TRUE,
                      widthConnectors = 0.5,
                      colConnectors = 'grey50',
                      gridlines.major = TRUE,
                      gridlines.minor = FALSE,
                      border = 'partial',
                      borderWidth = 1.5,
                      borderColour = 'black')
# p4

library(gridExtra)
library(grid)
png(filename ="./Results/Volcano/DEG_panel_600DPI.png", width = 10, height = 10, units = "in", pointsize = 36, res = 600)
grid.arrange(p1, p2, p3, p4,
             nrow = 2,
             ncol = 2
             # top = textGrob('EnhancedVolcano',
             #                just = c('center'),
             #                gp = gpar(fontsize = 32))
             )
dev.off()
