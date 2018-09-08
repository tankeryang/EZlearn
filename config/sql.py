READ_SQL = {
    'MMS': {
        'STORE_SC_WEEK': {
            'ALLOCATE': "",
            'TRANSFORM': """
                select * from ads_mms.training_data_store_sc_week_merge
                where tempreture_day_avg is not null
                and year_code >= '2017'
                and sales_qty <= 50
            """
        },
        'REGION_SKC_WEEK': {
            'ALLOCATE': "",
            'TRANSFORM': """
                select * from ads_mms.training_data_region_skc_week_merge
                where tempreture_day_avg is not null
                and interval_weeks_to_list > 4
                and interval_weeks_to_list <= 24
                and year_code >= '2017'
                and sales_qty <= 30
            """
        }
    }
}
