# -*- coding: UTF-8 -*-

# 模型使用特征
cols_feature = [
    #### order & context
    "has_been_rejected_once",
    "create_hour",
    "order_hour",
    "is_appointment",
    "order_distance",

    "order_type",
    "is_prepaid",
    "order_vehicle_id",
    "pickup_dist",
    "is_with_remark",

    "path_point_num",
    "is_cross_city",
    "follow_person_num_from_remark",
    "order_vehicle_name_enum",
    "city_id",

    "end_city_id",
    "length_remark",
    "create_hour_minute",
    "order_hour_minute",
    "total_price_fen",

    "tips_price_fen",
    "weekday",
    "is_special_request_cart",
    "is_special_request_photo_receipt",
    "is_special_request_paper_receipt",

    "is_special_request_move",
    "is_special_request_return",
    "order_is_request_tailgate",
    "order_is_request_seat_type",

    #### driver
    "driver_age",
    "driver_vip_level_enum",
    "total_pickup_num",
    "driver_avg_completed_transation_per_order",
    "driver_complete_rate",

    "be_favorite_rate",
    "define_vehicle_type_enum",
    "vehicle_type_enum",
    "driver_appointment_order_canceled_rate_90d",
    "driver_prepaid_order_canceled_rate_90d",

    "driver_cash_order_canceled_rate_90d",
    "driver_canceled_order_3q_pickup_duration_90d",
    "driver_completed_order_1q_pickup_distance_90d",
    "driver_immediate_order_canceled_rate_30d",
    "driver_prepaid_order_canceled_rate_30d",

    "driver_completed_order_avg_pickup_distance_30d",
    "driver_completed_order_2q_order_distance_14d",
    "driver_appointment_canceled_rate_180d",
    "driver_cash_canceled_rate_180d",
    "driver_is_brokerage_canceled_rate_180d",

    "vehicle_transport_type",
    "vehicle_power_type",
    "drv_respond_sum",
    "drv_veh_cancel_rate_0",  # 不同车型的历史取消率
    "drv_veh_cancel_rate_1",

    "drv_veh_cancel_rate_2",
    "drv_veh_cancel_rate_3",
    "drv_veh_cancel_rate_4",
    "drv_veh_cancel_rate_5",
    "drv_veh_cancel_rate_6",

    "drv_veh_cancel_rate_7",
    "drv_veh_cancel_rate_8",
    "drv_veh_cancel_rate_9",
    "drv_veh_cancel_rate_10",
    "drv_veh_cancel_rate_11",

    "drv_veh_rate_0",  # 不同车型的订单量占比
    "drv_veh_rate_1",
    "drv_veh_rate_2",
    "drv_veh_rate_3",
    "drv_veh_rate_4",

    "drv_veh_rate_5",
    "drv_veh_rate_6",
    "drv_veh_rate_7",
    "drv_veh_rate_8",
    "drv_veh_rate_9",

    "drv_veh_rate_10",
    "drv_veh_rate_11",
    "driver_90d_reject_reason1",  # from xuewei table
    "driver_90d_reject_reason2",
    "driver_90d_special_request_c",  # cart

    "driver_90d_special_request_d",  # photo
    "driver_90d_special_request_e",  # paper
    "driver_90d_special_request_f",  # move
    "driver_90d_special_request_g",  # return
    "driver_start_city_id",

    "order_pk_start_city_driver_major_city_order_rate",
    "driver_end_city_id",
    "order_pk_end_city_driver_major_city_order_rate",
    "vehicle_is_with_tailgate",  # 线上传司机画像即可 vehicle_ab_base
    "vehicle_seat_type_enum",

    "vehicle_height_limitation_metre",
    "vehicle_compartment_type_enum",
    "vehicle_roof_type_enum",
    "vehicle_is_steel_car",

    #### 10.3 user
    "member_no",
    "user_created_order_num2",
    "user_completed_rate2",
    "user_prepaid_order_canceled_rate_90d",
    "user_cash_order_canceled_rate_90d",

    "user_completed_order_1q_pickup_distance_90d",
    "user_completed_order_2q_order_distance_90d",
    "user_cash_canceled_rate_180d",
    "user_is_brokerage_canceled_rate_180d",
    "ib_order_veh_sum",

    "user_order_sv_ratio",
    "user_order_mv_ratio",
    "user_order_sc_ratio",
    "user_order_mc_ratio",
    "user_order_lc_ratio",

    "usr_veh_cancel_rate_0",
    "usr_veh_cancel_rate_1",
    "usr_veh_cancel_rate_2",
    "usr_veh_cancel_rate_3",
    "usr_veh_cancel_rate_4",

    "usr_veh_cancel_rate_5",
    "usr_veh_cancel_rate_6",
    "usr_veh_cancel_rate_7",
    "usr_veh_cancel_rate_8",
    "usr_veh_cancel_rate_9",

    "usr_veh_cancel_rate_10",
    "usr_veh_cancel_rate_11",
    "usr_veh_rate_0",
    "usr_veh_rate_1",
    "usr_veh_rate_2",

    "usr_veh_rate_3",
    "usr_veh_rate_4",
    "usr_veh_rate_5",
    "usr_veh_rate_6",
    "usr_veh_rate_7",

    "usr_veh_rate_8",
    "usr_veh_rate_9",
    "usr_veh_rate_10",
    "usr_veh_rate_11",
    "user_90d_reject_reason1",

    "user_90d_reject_reason2",
    "user_90d_special_request_c",  # cart
    "user_90d_special_request_d",  # photo
    "user_90d_special_request_e",  # paper
    "user_90d_special_request_f",  # move

    "user_90d_special_request_g",  # return
]


# 在模型特征的基础上，添加一些其他信息
cols_non_feature_info = ['order_id', 'ab_order', 'driver_id', 'user_id', 'dt', 'label']

# 新添加特征名称（待在模型中尝试)
cols_feature_more = []

cols_feature_all = cols_feature + cols_feature_more


