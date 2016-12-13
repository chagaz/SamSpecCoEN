# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# April 2016


import h5py
import numpy as np
import os
import pandas as pd
import pickle
import sys

sys.path.append('NormalisingData/projectX/')
from ExpressionDataset import ExpressionDataset
import mapData

def main():
    """ Pre-process E-MTAB-62 data into a .h5 file similar to those from ACES.

    Files generated
    ---------------
    Under 'ArrayExpress/postproc':
    hgu133a_rma_okFiles_080619_MAGETAB.dat:
        Normal (= control) data from E-MTAB-62, in .dat format.
        first line = gene names, space-separated, between " quotes (e.g. "1255_g_at").
        first column = GSM names, between " quotes (e.g. "GSM107074.CEL.gz").

    HG-U133A.pickle:
        Used to map Affymetrix U133A probe set IDs to Entrez gene IDs.

    MTAB-62_data.pickle:
        Normal (= control) data from E-MTAB-62, cleaned and mapped by mapData
        and transformed into an ExpressionDataset.
    
    MTAB-62.h5
        Normal (= control) data from E-MTAB-62, transformed into .h5 file.
    """
    res_dir = 'ArrayExpress/postproc'

    # Read E-MTAB-62 description file to extract normal samples
    fd = pd.read_csv('ArrayExpress/E-MTAB-62.sdrf.txt', delimiter='\t')
    normals = fd[fd['Characteristics[4 meta-groups]']=='normal']['Source Name'].tolist()

    csv_data_fname = 'ArrayExpress/hgu133a_rma_okFiles_080619_MAGETAB.csv'
    # first line = sample names, tab separated, between " quotes (e.g  "1102960569.CEL"  )
    # second line = to be ignored
    # first column = gene names, between " quotes (e.g. "1255_g_at")

    # Get sample IDs (first line of csv_data_fname)
    with open(csv_data_fname, 'r') as f:
        header = f.readline() # samples 
        f.close()
    cols = [i for (i, x) in enumerate(header.split("\t")) \
            if x.lstrip('"').rstrip('"') in normals]
    print len(cols), "samples"

    # Get gene names (first column of csv_data_fname)
    gene_names = np.loadtxt(csv_data_fname, 
                            usecols=[0], dtype='str')
    gene_names = gene_names[2:]
    print len(gene_names), "genes"

    # Get expression data itself
    data = np.loadtxt(csv_data_fname, 
                      skiprows=2, delimiter='\t', usecols=cols)

    
    

    # Transform hgu133a_rma_okFiles_080619_MAGETAB.csv into .dat
    # match the format of NormalisingData/R_playground/U133AnormalizedExpression.dat:
    # first line = gene names, space-separated, between " quotes (e.g. "1255_g_at")
    # first column = GSM names, between " quotes (e.g. "GSM107074.CEL.gz")
    datafile = '%s/hgu133a_rma_okFiles_080619_MAGETAB.dat' % res_dir

    fcol = np.array([header.split("\t")[ix] for ix in cols])
    fcol = fcol.reshape((fcol.shape[0], 1))
    D = np.hstack((fcol, data.T))

    # Save data into .dat file
    np.savetxt(datafile, D, fmt='%s', comments='',
               delimiter=" ", header=" ".join(['"%s"' % x for x in gene_names]))

    # Read the data as ExpressionDataset
    (affyIDs, patientIDs, exprMatrix, exprs) = mapData.readExpressionData(datafile)

    ### Create an ExpressionDataset object
    cohort_name = 'MTAB-62'
    ds = ExpressionDataset(cohort_name, exprMatrix, np.array(affyIDs), np.ones(len(patientIDs)),
                           np.array(patientIDs), checkNormalization=False, checkClassLabels=False)

    ### Map from Affymetrix probe set ID to Entrez gene ID
    print "Map AffyIDs to Entrez gene IDs"
    map_probe_ids_to_gene_ids = 'NormalisingData/projectX/HG-U133A.na32.annot.csv/HG-U133A.na32.annot.csv'
    map_fname = '%s/HG-U133A.pickle' % res_dir
    best_map_entrez_affy = mapData.ProbeToGeneID(map_probe_ids_to_gene_ids, ds, map_fname)
    best_map_affy_entrez = dict(zip(best_map_entrez_affy.values(), best_map_entrez_affy.keys()))

    idx = np.argwhere(np.in1d(ds.geneLabels, best_map_affy_entrez.keys()))[:, 0]

    ds_clean_probes = ds.extractGenesByIndices(('%s_clean_probes' % cohort_name), idx, 
                                              checkNormalization=False, checkClassLabels=False)

    print "Replace AffyIDs with gene IDs"
    for ix in range(len(ds_clean_probes.geneLabels)):
        ds_clean_probes.geneLabels[ix] = best_map_affy_entrez[ds_clean_probes.geneLabels[ix]]

    print "Mean-center"
    ds_clean_probes.expressionData = ds_clean_probes.expressionData - ds_clean_probes.expressionData.mean(0)

    ### Dump clean data
    ds_clean_probes = ExpressionDataset(cohort_name, ds_clean_probes.expressionData,
                                        ds_clean_probes.geneLabels,
                                        ds_clean_probes.patientClassLabels, 
                                        ds_clean_probes.patientLabels, checkNormalization=False,
                                        checkClassLabels=False)
    pickle.dump(ds_clean_probes, open('%s/%s_data.pickle' % (res_dir, cohort_name), 'wb'))

    ### Convert pickle data into hdf5 format
    from ConvertPickleData import EmitDataset

    data_clean = pickle.load(open('%s/%s_data.pickle' % (res_dir, cohort_name)))
    
    hdf5_fname = '%s/%s.h5' % (res_dir, cohort_name)
    hdf5_f = h5py.File(hdf5_fname, 'w')
    EmitDataset(data_clean, None, hdf5_f)
    hdf5_f.close()
    
    
if __name__ == "__main__":
    main()