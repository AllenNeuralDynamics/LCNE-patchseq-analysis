"""Utilities for querying the LIMS database.

From Brian
"""

import json

import pandas as pd  # pandas will be needed to work in a dataframe
import pg8000  # pg8000 access SQL databases

# code from Agata
# these are nice functions to open LIMS, make a query and then close LIMS after


def _connect(user, host, database, password, port):
    """Connect to the database"""
    conn = pg8000.connect(user=user, host=host, database=database, password=password, port=port)
    return conn, conn.cursor()


def _select(cursor, query):
    cursor.execute(query)
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, c)) for c in cursor.fetchall()]


def limsquery(query, user, host, database, password, port):
    """A function that takes a string containing a SQL query, connects to the LIMS database
    and outputs the result."""
    conn, cursor = _connect(user, host, database, password, port)
    try:
        results = _select(cursor, query)
    finally:
        cursor.close()
        conn.close()
    return results


# this last function will take our query results and put them in a dataframe
# so that they are easy to work with
def get_lims_dataframe(query):
    """Return a dataframe with lims query"""

    # Get credentials from json
    with open("LIMS_credentials.json") as f:
        credentials = json.load(f)

    result = limsquery(query, **credentials)
    try:
        data_df = pd.DataFrame(data=result, columns=result[0].keys())
    except IndexError:
        print("Could not find results for your query.")
        data_df = pd.DataFrame()
    return data_df


# Query for LCNE patchseq experiments
def get_lims_LCNE_patchseq():
    """Code from Brian"""
    lims_query = """
        SELECT 
            s.name as specimen_name,
            s.id as specimen_id,
            err.id AS ephys_roi_id,
        CASE d.gender_id
            WHEN 1 THEN 'male'
            WHEN 2 THEN 'female'
            ELSE 'unknown'
        END as gender, d.date_of_birth, a.name as age, proj.code, err.recording_date,
        err.workflow_state as Ephys_QC, s.patched_cell_container, err.storage_directory
        FROM donors d
        JOIN ages a ON a.id=d.age_id
        LEFT JOIN specimens s ON s.donor_id = d.id
        LEFT JOIN ephys_roi_results err ON s.ephys_roi_result_id = err.id
        JOIN projects proj ON s.project_id = proj.id
        WHERE proj.code in ('mIVSCC-MET-R01_LC')
        AND err.workflow_state IS NOT NULL    
        """
    lims_df = get_lims_dataframe(lims_query)
    return lims_df


if __name__ == "__main__":
    lims_df = get_lims_LCNE_patchseq()
    print(lims_df.head())
