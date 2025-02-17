def _clang_format_test_impl(ctx):
    sources = ctx.files.srcs
    clang_format_file = ctx.file.clang_format_file
    out = ctx.actions.declare_file(ctx.label.name + ".sh")

    script_content = """
#!/bin/bash
set -e
NEEDS_FORMATTING=0
CLANG_FORMAT="/usr/bin/clang-format-15"
CLANG_FORMAT_FILE="{clang_format_file}"

echo "Debug: Current working directory is $(pwd)"
echo "Debug: Content of current directory:"
ls -la
echo "Debug: CLANG_FORMAT_FILE is set to $CLANG_FORMAT_FILE"
echo "Debug: Does the file exist? $(test -f "$CLANG_FORMAT_FILE" && echo "Yes" || echo "No")"

if ! command -v $CLANG_FORMAT &> /dev/null; then
    echo "clang-format could not be found. Please install it."
    exit 1
fi

""".format(clang_format_file = clang_format_file.short_path)

    for src in sources:
        script_content += """
REPLACEMENTS=$($CLANG_FORMAT -output-replacements-xml --style=file:$CLANG_FORMAT_FILE {src})
if echo "$REPLACEMENTS" | grep -q "<replacement "; then
    echo "$CLANG_FORMAT -i --style=file:$CLANG_FORMAT_FILE {src}"
    NEEDS_FORMATTING=1
fi
""".format(src = src.short_path)

    script_content += """
if [ $NEEDS_FORMATTING -eq 0 ]; then
    echo "All files are correctly formatted."
else
    echo "Some files need formatting. Run the commands above to fix."
fi
exit $NEEDS_FORMATTING
"""

    ctx.actions.write(
        output = out,
        content = script_content,
        is_executable = True,
    )

    return [DefaultInfo(
        executable = out,
        runfiles = ctx.runfiles(files = sources + [clang_format_file]),
    )]

clang_format_test = rule(
    implementation = _clang_format_test_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = [".cpp", ".h", ".hpp", ".cu"], mandatory = True),
        "clang_format_file": attr.label(allow_single_file = True, mandatory = True),
    },
    test = True,
)