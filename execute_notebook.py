"""Executes sentiment_analysis.ipynb sequentially, capturing text output and matplotlib plots directly into the notebook cells."""
import sys
import os
import io
import base64
import traceback
from contextlib import redirect_stdout, redirect_stderr
import nbformat as nbf
import matplotlib.pyplot as plt

def run_notebook(notebook_path: str = 'sentiment_analysis.ipynb'):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    exec_globals = {
        '__name__': '__main__',
        '__file__': os.path.abspath(notebook_path)
    }

    execution_count = 1
    total_code_cells = sum(1 for c in nb.cells if c.cell_type == 'code')
    print(f"Starting execution of {total_code_cells} code cells in {notebook_path}...")

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue

        print(f"\n[{execution_count}/{total_code_cells}] Executing code cell (Line ~{idx})...")
        cell_source = cell.source

        # Buffer for stdout and stderr
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        # Clear any pre-existing plots
        plt.close('all')

        cell.outputs = []
        cell.execution_count = execution_count

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(cell_source, exec_globals)
            success = True
        except Exception as e:
            success = False
            err_msg = traceback.format_exc()
            stderr_buf.write(err_msg)
            print(f"ERROR in cell {execution_count}: {e}")

        # Collect captured text output
        out_text = stdout_buf.getvalue()
        err_text = stderr_buf.getvalue()

        if out_text:
            cell.outputs.append(nbf.v4.new_output(
                output_type='stream',
                name='stdout',
                text=out_text
            ))
            # Also print to terminal console
            sys.stdout.write(out_text)

        if err_text:
            cell.outputs.append(nbf.v4.new_output(
                output_type='stream',
                name='stderr',
                text=err_text
            ))
            sys.stderr.write(err_text)

        # Collect any generated matplotlib figures
        fig_nums = plt.get_fignums()
        for fnum in fig_nums:
            fig = plt.figure(fnum)
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=110)
            img_buf.seek(0)
            img_b64 = base64.b64encode(img_buf.read()).decode('utf-8')
            cell.outputs.append(nbf.v4.new_output(
                output_type='display_data',
                data={'image/png': img_b64, 'text/plain': '<Figure size ...>'}
            ))
        plt.close('all')

        execution_count += 1

        if not success:
            print(f"Halting execution due to error in cell {execution_count - 1}.")
            break

    # Save executed notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"\nExecution finished! Notebook saved to {notebook_path}.")

if __name__ == '__main__':
    run_notebook()
