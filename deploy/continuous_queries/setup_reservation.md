# Enterprise Reservation Setup for Continuous Queries

BigQuery continuous queries require an Enterprise or Enterprise Plus
reservation. This guide covers the minimum setup.

## 1. Create a reservation

```bash
bq mk \
  --reservation \
  --project_id=PROJECT \
  --location=REGION \
  --edition=ENTERPRISE \
  --slot_capacity=100 \
  continuous-query-reservation
```

## 2. Create an assignment

Assign the reservation to the project (or a specific dataset):

```bash
bq mk \
  --reservation_assignment \
  --project_id=PROJECT \
  --location=REGION \
  --reservation_id=continuous-query-reservation \
  --assignee_id=PROJECT \
  --assignee_type=PROJECT \
  --job_type=CONTINUOUS
```

## 3. Verify

```bash
bq show --reservation --project_id=PROJECT --location=REGION \
  continuous-query-reservation
```

## 4. Run a continuous query

```bash
bq query --use_legacy_sql=false --continuous=true < session_scoring.sql
```

## 5. Monitor running jobs

```sql
SELECT
  job_id,
  state,
  creation_time,
  TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), creation_time, SECOND) AS running_seconds,
  total_bytes_processed
FROM
  `region-REGION`.INFORMATION_SCHEMA.JOBS
WHERE
  job_type = 'QUERY'
  AND state = 'RUNNING'
  AND configuration.query.continuous = true
ORDER BY
  creation_time DESC;
```

## Cost estimate

Enterprise reservations are billed per slot-hour. 100 slots at the
Enterprise tier costs approximately $0.06/slot-hour = $6.00/hour.
Adjust `slot_capacity` based on query complexity and throughput.

## Cleanup

```bash
bq rm --reservation_assignment \
  --project_id=PROJECT --location=REGION \
  continuous-query-reservation.PROJECT

bq rm --reservation \
  --project_id=PROJECT --location=REGION \
  continuous-query-reservation
```
