import sqlite3

conn = sqlite3.connect(r'db/resistance_genes.db')
cursor = conn.cursor()

# Get table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('=== TABLES ===')
for t in cursor.fetchall():
    print(t[0])

# Get schema
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
print('\n=== SCHEMA ===')
for s in cursor.fetchall():
    print(s[0])

# Count rows
cursor.execute('SELECT COUNT(*) FROM sequences')
print(f'\n=== TOTAL ROWS: {cursor.fetchone()[0]} ===')

# Show all entries with lengths and validation
cursor.execute('SELECT gene_name, dna_sequence FROM sequences ORDER BY gene_name')
rows = cursor.fetchall()

valid_chars = set('ATCGN')
issues = []

print(f'\n{"GENE NAME":<35} {"LEN":>6}  {"STATUS":<20}  FIRST 60 CHARS')
print('-'*120)
for gene, seq in rows:
    clean = ''.join(seq.upper().split()) if seq else ''
    invalid = set(clean) - valid_chars
    if not clean:
        status = '*** EMPTY/NULL'
        issues.append((gene, 'EMPTY/NULL', ''))
    elif invalid:
        status = f'*** BAD CHARS: {invalid}'
        issues.append((gene, f'INVALID CHARS {invalid}', clean[:60]))
    else:
        status = 'OK'
    print(f'{gene:<35} {len(clean):>6}  {status:<20}  {clean[:60]}')

print(f'\n=== SUMMARY ===')
print(f'Total genes: {len(rows)}')
print(f'Issues found: {len(issues)}')
if issues:
    print('\nPROBLEMATIC ENTRIES:')
    for g, reason, seq in issues:
        print(f'  [{reason}] {g}: {seq}')

conn.close()
