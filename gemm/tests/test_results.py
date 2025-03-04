from gemm import Results

def test_results():
    results = Results('bGemm.json')
    print(results.df)
    assert True
