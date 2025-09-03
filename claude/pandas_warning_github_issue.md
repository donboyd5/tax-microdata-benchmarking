# GitHub Issue: Fix Pandas FutureWarning for ChainedAssignmentError

## Title
Fix pandas ChainedAssignmentError warning in create_taxcalc_growth_factors.py

## Labels
- bug
- maintenance
- pandas-compatibility

## Issue Body

### Description
A `FutureWarning` appears when running `make data` on some systems, warning about chained assignment that will break in pandas 3.0. This needs to be fixed to ensure future compatibility.

### Current Behavior
When running `make clean; make data`, the following warning appears (on some systems):

```
python tmd/create_taxcalc_growth_factors.py
/home/don-boyd/Documents/psl/tax-microdata-benchmarking/tmd/create_taxcalc_growth_factors.py:78: 
FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
You are setting values through chained assignment. Currently this works in certain cases, 
but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) 
this will never work to update the original DataFrame or Series, because the intermediate 
object on which we are setting values will behave as a copy.

A typical example is when you are setting values in a column of a DataFrame, like:
df["col"][row_indexer] = value

Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single 
step and ensure this keeps updating the original `df`.

  added.YEAR.iat[idx] = last_puf_year + idx + 1
```

### Expected Behavior
The code should run without warnings and be compatible with pandas 3.0 when it's released.

### Root Cause
The issue is at line 78 of `tmd/create_taxcalc_growth_factors.py`:
```python
added.YEAR.iat[idx] = last_puf_year + idx + 1
```

This uses **chained assignment** where:
1. `added.YEAR` creates an intermediate Series object
2. `.iat[idx]` attempts to modify that intermediate object
3. With pandas 3.0's Copy-on-Write default, this modification won't propagate back to the original DataFrame

### Proposed Solution
Replace the chained assignment with direct assignment using `.at`:

```python
# Current (line 78):
added.YEAR.iat[idx] = last_puf_year + idx + 1

# Fixed:
added.at[idx, 'YEAR'] = last_puf_year + idx + 1
```

**Important**: Use `.at` not `.loc` - the `.loc` accessor can create new rows if the index doesn't exist, while `.at` works exactly like `.iat` but without chained assignment.

### Impact Assessment
- **Current Impact**: Warning only, functionality still works
- **Future Impact**: Will break when pandas 3.0 is released
- **Fix Compatibility**: The proposed fix is backwards compatible with all pandas versions

### Testing Performed
- [ ] Verified the fix eliminates the warning
- [ ] Confirmed output remains identical with the fix
- [ ] Tested with both pandas 1.x and 2.x versions

### System Information
The warning appears inconsistently across systems, likely due to:
- Different pandas versions (newer versions ≥1.5.0 show the warning more prominently)
- Different Python warning configurations
- Different virtual environment setups

### Steps to Reproduce
1. Clone the repository
2. Set up environment: `pip install -e .`
3. Run: `make clean; make data`
4. Observe warning during `python tmd/create_taxcalc_growth_factors.py` execution

### Additional Context
- There may be other instances of chained assignment in the codebase that should be reviewed
- Enabling pandas Copy-on-Write mode in development (`pd.options.mode.copy_on_write = True`) would help catch these issues early

### Checklist
- [ ] I have tested the proposed fix
- [ ] The fix maintains identical output
- [ ] No new warnings are introduced
- [ ] Tests pass with the change

---

### Quick Fix for Reviewers
The one-line change needed:
```diff
- added.YEAR.iat[idx] = last_puf_year + idx + 1
+ added.at[idx, 'YEAR'] = last_puf_year + idx + 1
```

Location: `tmd/create_taxcalc_growth_factors.py`, line 78