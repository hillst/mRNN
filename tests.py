from fasta import load_fasta, tokenize_dna
def load_test_seqs():
    seqs = load_fasta("resources/test.fa")
    return seqs
