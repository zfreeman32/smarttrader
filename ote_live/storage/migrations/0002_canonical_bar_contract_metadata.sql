ALTER TABLE canonical_bars
    ADD COLUMN symbol TEXT;

ALTER TABLE canonical_bars
    ADD COLUMN contract_symbol TEXT;

ALTER TABLE canonical_bars
    ADD COLUMN instrument_id INTEGER;
