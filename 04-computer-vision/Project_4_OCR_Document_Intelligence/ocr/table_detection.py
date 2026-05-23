import camelot


# Extract tables
tables = camelot.read_pdf(
    "data/pdfs/sample.pdf",
    pages="1"
)


print(
    f"Found {tables.n} tables"
)


# Export first table
tables[0].to_csv(
    "outputs/table.csv"
)


print("Table saved.")