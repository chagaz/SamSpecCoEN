# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr

import argparse
import os
import sys

mapping_f = '../NormalisingData/projectX/HG-U133A.na32.annot.csv/HG-U133A.na32.annot.csv'


def main():
    """ Convert a list of Entrez Gene IDs in a list of gene symbols.
    
    Example
    -------
        $ python map_Entrez_to_gene_symbol.py ../outputs/U133A_combat_RFS/KEGG_edges1210/su
btype_stratified/repeat0/results/nodes/final_selection_genes.txt ../outputs/U133A_combat_RFS/KEGG_edges1210/su
btype_stratified/repeat0/results/nodes/final_selection_genes_symbols.txt
    
    """
    parser = argparse.ArgumentParser(description="Convert EntrezID to Gene Symbol",
                                     add_help=True)
    parser.add_argument("entrez_path", help="File containing the Entrez IDs")
    parser.add_argument("symbol_path", help="File containing the Gene Symbols (to write)")
    args = parser.parse_args()

    # Create mapping dictionary
    mapping_dict = {} # EntrezID:GeneSymbol
    with open(mapping_f) as f:
        header = f.readline()
        while header.startswith('#'):
            header = f.readline()

        header = header.split("\",\"")
        # print "Columns in mapping file: "
        # print header

        entrez_idx = header.index("Entrez Gene")
        symbol_idx = header.index("Gene Symbol")
        
        while True:
            line = f.readline()
            if len(line) > 0:
                line = line.split("\",\"")
                try:
                    entrez_id = line[entrez_idx].replace("\"", "")
                    if entrez_id == "42":
                        for i, x in enumerate(line):
                            print i, x
                        sys.exit(0)
                    symbol_id = line[symbol_idx].replace("\"", "")
                except:
                    continue
                mapping_dict[entrez_id] = symbol_id
            else:
                break
        f.close()    

        
    # Convert file
    error_cnt = 0
    with open(args.entrez_path) as f:
        with open(args.symbol_path, 'w') as g:
            for line in f:
                entrez_id = line.split()[0].split("_")[1]
                try:
                    symbol_id = mapping_dict[entrez_id]
                    g.write("%s\n" % symbol_id)
                except KeyError:
                    error_cnt += 1
                    sys.stderr.write("Could not find Gene Symbol for Entrez ID %s.\n" % entrez_id)
            g.close()
        f.close()

    print error_cnt, "genes could not be mapped."
            
if __name__ == "__main__":
    main()