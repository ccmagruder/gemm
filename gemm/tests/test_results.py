from gemm import Results


def test_results():
    results = Results("bGemm.json", "n")
    results.plot(x="n", ref_exponent=2)
