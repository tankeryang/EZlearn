SQL_SELECT_FROM_ALLOCATE_FETURE = """
    select
        city_name,
        week_code,
        area_code,
        store_type,
        store_level,
        store_retail_amount_mean_8weeks,
        store_sales_amount_mean_8weeks,
        sub_cate,
        leaf_cate,
        color_code,
        lining,
        customer_level_1_proportion_last_week,
        customer_level_2_proportion_last_week,
        customer_level_3_proportion_last_week,
        special_day_type,
        weather_day_most,
        weather_night_most,
        tempreture_day_highest,
        tempreture_day_avg,
        tempreture_day_lowest,
        tempreture_day_gap,
        tempreture_night_highest,
        tempreture_night_avg,
        tempreture_night_lowest,
        tempreture_night_gap,
        tempreture_avg_gap,
        retail_amount_mean,
        retail_amount_mean_gap_with_store,
        sales_amount_mean,
        sales_amount_mean_gap_with_store
    from ods_predict.flask_test_allocate
    where store_code = '{store_code}'
    and skc_code = '{skc_code}'
"""


SQL_SELECT_FROM_TRANSFORM_FETURE = """
    select
        week_code,
        interval_weeks_to_prev,
        interval_weeks_to_list,
        city_name,
        area_code,
        store_type,
        store_level,
        store_retail_amount_mean_8weeks,
        store_sales_amount_mean_8weeks,
        sub_cate,
        leaf_cate,
        color_code,
        lining,
        customer_level_1_proportion_last_week,
        customer_level_2_proportion_last_week,
        customer_level_3_proportion_last_week,
        special_day_type,
        weather_day_most,
        weather_night_most,
        tempreture_day_highest,
        tempreture_day_avg,
        tempreture_day_lowest,
        tempreture_day_gap,
        tempreture_night_highest,
        tempreture_night_avg,          
        tempreture_night_lowest
        tempreture_night_gap,
        tempreture_avg_gap,
        retail_amount_mean,
        retail_amount_mean_gap_with_store,
        sales_amount_mean,
        sales_amount_mean_gap_with_store,
        discount_rate_mean_last_4week,
        discount_rate_mean_last_3week,
        discount_rate_mean_last_2week,
        discount_rate_mean_last_week,
        discount_rate_mean,
        discount_rate_mean_change_rate,
        skc_con_sale_rate_last_week,
        sales_qty_last_2week,
        sales_qty_last_week,
        sales_qty_last_week_and_last_2week_gap
    from ods_predict.flask_test_transform
    where store_code = '{store_code}'
    and skc_code = '{skc_code}'
"""