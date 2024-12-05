-- models/customer_insights.sql
-- Description: Aggregated customer insights
-- Source: customer_feedback (referenced via Jinja `ref`)

SELECT
    customer_id,
    synthesize("feedback from a customer: {feedback} and the time stamp from when the feedback was received: {timestamp}", "analyst", "feedback counts")
FROM {{ ref('customer_feedback') }}
GROUP BY customer_id;