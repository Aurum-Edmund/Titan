import subprocess, sys, json, os, pathlib
def run(cmd):
    print(">", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode!=0:
        print(r.stderr); sys.exit(r.returncode)

def main():
    # 1) tokenizer exists
    assert pathlib.Path("tokenizer_local").exists(), "tokenizer_local missing"
    # 2) specials unit
    run([sys.executable, "-m", "pytest", "tests/test_tokenizer_specials.py","-q"])
    # 3) generate data
    run([sys.executable, "scripts/gen_echo_dataset.py","--total","50000"])
    # 4) dataloader pad/ignore_index
    run([sys.executable, "-m", "pytest", "tests/test_dataloader_padding.py","-q"])
    # 5) math core sanity
    run([sys.executable, "-m", "pytest", "tests/test_math_core.py","-q"])
    print("✅ kit check passed")

if __name__=="__main__":
    main()
