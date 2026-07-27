#!/usr/bin/env python3
"""Route hash_product's device allocations through the device buffer pool.

Transforms spgemm_kernel_hash.cu:
  1. dev_pool_reset() inserted right after the HashProf prof(...) line.
  2. CHECK_CUDA(cudaMalloc(&VAR, EXPR));  ->  VAR = decltype(VAR)(dev_alloc(EXPR));
     (EXPR captured with balanced-paren scanning; dev_alloc handles its own errors)
  3. cudaFree(X)  ->  dev_free(X)   (all sites; pool mode = no-op, reset reclaims)

Idempotent: if already wired (dev_pool_reset present), exits without changes.
"""
import re, sys, shutil

PATH = "src/spgemm_kernel_hash.cu"

def main():
    src = open(PATH).read()
    if "dev_pool_reset()" in src:
        print("already wired (dev_pool_reset present) — no change")
        return

    shutil.copyfile(PATH, PATH + ".bak_devpool")

    # 1) insert dev_pool_reset() after HashProf prof(...) in hash_product.
    #    The HashProf line: `HashProf prof(att ? "atth-prof" : "hash-prof");`
    m = re.search(r'(    HashProf prof\(att \? "atth-prof" : "hash-prof"\);\n)', src)
    assert m, "HashProf prof line not found"
    src = src[:m.end()] + "    dev_pool_reset();   // device arena:本调用所有 device buffer 复用(省 ~13 cudaMalloc/Free)\n" + src[m.end():]

    # 2) replace CHECK_CUDA(cudaMalloc(&VAR, EXPR)); with VAR = decltype(VAR)(dev_alloc(EXPR));
    out = []
    i = 0
    pat = "CHECK_CUDA(cudaMalloc(&"
    n_malloc = 0
    while True:
        j = src.find(pat, i)
        if j < 0:
            out.append(src[i:]); break
        out.append(src[i:j])
        k = j + len(pat)
        # VAR = identifier
        mvar = re.match(r"[A-Za-z_]\w*", src[k:])
        assert mvar, f"no VAR after & at {k}"
        var = mvar.group(0); k += len(var)
        assert src[k] == ',', f"expected , after VAR {var} at {k}: {src[k:k+20]!r}"
        k += 1
        if src[k] == ' ': k += 1
        # EXPR = balanced parens until the matching ')' of cudaMalloc
        depth = 1  # we're inside cudaMalloc(
        start = k
        while k < len(src) and depth > 0:
            c = src[k]
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            k += 1
        assert depth == 0, "unbalanced parens in cudaMalloc EXPR"
        expr = src[start:k-1]   # exclude the closing ')'
        # now src[k] should be ')' (close CHECK_CUDA) then ';'
        assert src[k:k+2] == ");", f"expected ); after EXPR at {k}: {src[k:k+8]!r}"
        rep = f"{var} = decltype({var})(dev_alloc({expr}));"
        out.append(rep)
        i = k + 2   # skip ");"
        n_malloc += 1
    src = "".join(out)

    # 3) cudaFree( -> dev_free(   (all remaining sites)
    n_free = src.count("cudaFree(")
    src = src.replace("cudaFree(", "dev_free(")

    open(PATH, "w").write(src)
    print(f"wired: {n_malloc} cudaMalloc -> dev_alloc, {n_free} cudaFree -> dev_free")
    print("backup: src/spgemm_kernel_hash.cu.bak_devpool")

if __name__ == "__main__":
    main()
