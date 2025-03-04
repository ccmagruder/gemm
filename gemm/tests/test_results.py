from gemm import Results


def test_results():
    results = Results("bGemm.json", "n")
    print(results.df)
    assert True
