# Import Bulk dataframe
#install.packages("readxl")
library("readxl")
library(RColorBrewer)
library("pheatmap")

# import BULK DF
# path <- "/Users/cleliacorridori/Dropbox_2021 Dropbox/Jorah Mormont/GRN_Inference/DATA/input_data/geni_IMP.xlsx"
path <- "/Users/cleliacorridori/Dropbox_2021 Dropbox/Jorah Mormont/GRN_Inference/DATA/input_data/imp_genes_bulk_right.csv"

geneExp <- read.csv(path)[-1]

# Extract just the numeric data into a matrix with named rows by gene
geneExp <- as.data.frame(geneExp)
rownames(geneExp) <- geneExp$Symbol

# matrix format
geneExp_matrix <- as.matrix(geneExp[3:34])

# divide 21 and 2iL
geneExp_matrix_2iL <- as.matrix(geneExp[3:18])
geneExp_matrix_2i <- as.matrix(geneExp[19:34])

# heatmap(geneExp_matrix_2i, Rowv=NA,
#         Rowv=NA, col=rev(brewer.pal(9,"RdBu")))

# Z-SCORE
# pheatmap(geneExp_matrix_2i, scale="row", 
#          cluster_cols = FALSE, cluster_rows = FALSE, 
#          main="Bulk - 2i", 
#          fontsize = 12, fontsize_row = 12, fontsize_col = 12,)

# LOG2(CPM+1) 
# ????? era questa la normalizzazione del dataset?
# pheatmap(log2(geneExp_matrix_2i+1), 
#          cluster_cols = FALSE, cluster_rows = TRUE, 
#          main="Bulk - 2i")

# Mean of the two sample per each point
geneExp.mean <- (geneExp[,seq(3,34,2)]+geneExp[,seq(3,34,2)])/2
# divide 21 and 2iL
geneExp_matrix_2iL.mean <- as.matrix(geneExp.mean[1:8])
geneExp_matrix_2i.mean <- as.matrix(geneExp.mean[9:16])
colnames(geneExp_matrix_2iL.mean) <- c("2i+LIF", "24h", "36h","48h","60h", "72h", "84h", "96h")

pheatmap(geneExp_matrix_2i.mean, scale="row", 
         cluster_cols = FALSE, cluster_rows = FALSE, 
         main="Bulk - 2i", 
         fontsize = 10, fontsize_row = 9, fontsize_col = 9,)

pheatmap(geneExp_matrix_2iL.mean, scale="row", 
         cluster_cols = FALSE, cluster_rows = FALSE, 
         main="Bulk - 2i+LIF", 
         fontsize = 10, fontsize_row = 9, fontsize_col = 9,)
