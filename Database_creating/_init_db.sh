#!/usr/bin/env bash
# Reset and repopulate the two tables
# Usage:  ./_init_db.sh | psql -v ON_ERROR_STOP=1 -U reffie_user -h localhost -d reffie_homework

DROP TABLE IF EXISTS listing  CASCADE;
DROP TABLE IF EXISTS properties CASCADE;


CREATE TABLE properties (
    property_id      UUID PRIMARY KEY,
    team_id          UUID,
    street_part      TEXT,
    unit_part        TEXT,
    city             TEXT,
    state            TEXT,
    zipcode          TEXT,
    full_address     TEXT,
    token_set        TEXT,    
    type_norm        TEXT
);

CREATE TABLE listing (
    listing_id       UUID PRIMARY KEY,
    property_id      UUID,
    team_id          UUID,
    street_part      TEXT,
    unit_part        TEXT,
    city             TEXT,
    state            TEXT,
    zipcode          TEXT,
    full_address     TEXT,
    token_set        TEXT
);


\copy properties(property_id, team_id, street_part, unit_part, city, state, zipcode,full_address, token_set, type_norm)FROM './properties_cleaned.csv'WITH (FORMAT csv, HEADER true);

\copy listing(listing_id, property_id, team_id, street_part, unit_part, city,state, zipcode, full_address, token_set)FROM './listings_cleaned.csv'WITH (FORMAT csv, HEADER true, NULL '');
