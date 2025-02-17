def _buildifier_test_impl(ctx):
    buildifier = ctx.executable._buildifier
    out = ctx.actions.declare_file(ctx.label.name + ".sh")

    # Create a space-separated list of input files
    input_files = " ".join([f.short_path for f in ctx.files.srcs])

    ctx.actions.write(
        out,
        content = """
            #!/bin/bash
            set -euo pipefail
            
            BUILDIFIER="{buildifier}"
            if [ ! -f "$BUILDIFIER" ]; then
                echo "Error: buildifier executable not found at $BUILDIFIER"
                exit 1
            fi
            
            echo "Found buildifier at: $BUILDIFIER"
            echo "Buildifier version:"
            $BUILDIFIER --version || true
            
            echo "Files to check:"
            echo "{input_files}"
            
            FAILED_FILES=()
            for FILE in {input_files}; do
                if ! $BUILDIFIER --mode=diff "$FILE" > /dev/null 2>&1; then
                    FAILED_FILES+=("$FILE")
                fi
            done

            if [ ${{#FAILED_FILES[@]}} -ne 0 ]; then
                echo "Buildifier found formatting issues in the following files:"
                for FILE in "${{FAILED_FILES[@]}}"; do
                    echo "  $FILE"
                    echo "To fix, run: ./bazel-bin/csdecomp/tests/buildifier_test.sh.runfiles/com_github_bazelbuild_buildtools/buildifier/buildifier_/buildifier --mode=fix $FILE"
                done
                exit 1
            fi

            echo "All files pass buildifier checks."
            exit 0
        """.format(
            buildifier = buildifier.short_path,
            input_files = input_files,
        ),
        is_executable = True,
    )
    runfiles = ctx.runfiles(files = [buildifier] + ctx.files.srcs)
    return [DefaultInfo(
        runfiles = runfiles,
        executable = out,
    )]

buildifier_test = rule(
    implementation = _buildifier_test_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "_buildifier": attr.label(
            default = "@com_github_bazelbuild_buildtools//buildifier",
            executable = True,
            cfg = "host",
        ),
    },
    test = True,
)
