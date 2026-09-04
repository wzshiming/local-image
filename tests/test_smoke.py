def test_import_and_version():
    import flux_server

    assert flux_server.__version__ == "0.1.0"
