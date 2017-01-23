# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr

import argparse
import os
import sys

# Total number of genes used
# Nodes: 12750
# KEGG_edges1210: 3238

import scipy.stats as st


def main():
    """ Compare a list of selected genes to the reference sets of genes.

    Created file
    ------------
    <results_path>/enrichment9.txt:
        For each reference set, p-value of enrichment for the selected genes.
        i.e. Probability that the selected genes set containts at least as many reference genes
        as observed.
    
    Example
    -------
        $ python compare_genesets.py \
    ../outputs/U133A_combat_RFS/KEGG_edges1210/subtype_stratified/repeat0/results/regline/ \
    ../FERAL_supp_data/Allahyar.285.sup.1 \
    -l ../outputs/U133A_combat_RFS/KEGG_edges1210/genes_in_network_GeneSymbols.txt
    
        $ python compare_genesets.py \
    ../outputs/U133A_combat_RFS/KEGG_edges1210/subtype_stratified/repeat0/results/nodes/ \
    ../FERAL_supp_data/Allahyar.285.sup.1 -n 12750
    """
    parser = argparse.ArgumentParser(description="Compare selected genes to given gene set.",
                                     add_help=True)
    parser.add_argument("results_path", help="Folder containing results.")
    parser.add_argument("geneset_path", help="Folder containing files of Gene Symbols of reference gene sets.")
    parser.add_argument("-n", "--num_of_genes", type=int,
                        help="Total number of genes.")
    parser.add_argument("-l", "--lst_of_genes",
                        help='File containing total list of Gene Symbols')
    args = parser.parse_args()

    # Total size of universe
    if not args.num_of_genes:
        if not args.lst_of_genes:
            sys.stderr.write("args.num_of_genes or args.lst_of_genes must be provided.\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)
    
    # Read set of selected genes
    selects_path = "%s/final_selection_genes_symbols.txt" % args.results_path
    with open(selects_path) as f:
        selects_set = set([line.split()[0] for line in f])
        f.close()
    selects_size = len(selects_set)
    print selects_size, "genes selected."

    # Read total number/set of genes
    if args.lst_of_genes:
        if args.num_of_genes:
            sys.stderr.write("args.lst_of_genes provided; args.num_of_genes will be ignored.\n")
        with open(args.lst_of_genes) as f:
            total_set = set([line.split()[0] for line in f])
            f.close()
        num_of_genes = len(total_set)
    else:
        num_of_genes = args.num_of_genes

    # For each reference set of genes
    pval_dict = {} # reference_set_name:pvalue
    for geneset_name in os.listdir(args.geneset_path):
        geneset_file = "%s/%s" % (args.geneset_path, geneset_name)
        
        with open(geneset_file) as f:
            geneset_set = set([line.split()[0] for line in f])
            f.close()
        if total_set:
            # Restrict reference set of genes to those in the universe
            geneset_set = geneset_set & total_set
        geneset_size = len(geneset_set)
        # print geneset_size, "genes in reference gene set."

        # Fraction of genes from geneset_set in selects_set
        x = len(selects_set & geneset_set)
        # print x, "of the selected genes belong to the reference gene set."
        # obs = float(x)/float(geneset_size)
        # print "%.2f of selected genes belong to reference set." % obs

        # Hypergeometric test
        # M = num_of_genes
        # n = geneset_size
        # N = selects_size
        rv = st.hypergeom(num_of_genes, geneset_size, selects_size)
        # pval = rv.pmf(x)
        # print "Probability of selecting exactly %d genes from the gene set by chance: %.2e" % (x, pval)
        pval = 1 - rv.cdf(x - 1)
        # print "Probability of selecting at least %d genes from the reference set by chance: %.2e" % (x, pval)
        pval_dict[geneset_name.split(".")[0]] = pval

    # Write out results
    with open("%s/enrichment9.txt" % args.results_path, 'w') as f:
        for geneset_name, pval in pval_dict.iteritems():
            f.write("%s\t%.2e\n" % (geneset_name, pval))
        f.close()
    
if __name__ == "__main__":
    main()