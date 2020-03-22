#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import pandas

############## PROBLEM SPECIFIC FUNCTIONS ###########################
def MaxDueInstallmentsMinPeriod(data_transactions):
    import pandas
    #Creating table with product id, first installment period adn max due installements.
    max_due_install_min_period = pandas.pivot_table(data = data_transactions, index = 'aid',
                                            values = ['due_installments', 'period'], aggfunc = ['max', 'min'])
    max_due_install_min_period.columns = ['max_due_installments', 'max_period', 'min_due_installments', 'min_period']
    max_due_install_min_period.drop(['max_period', 'min_due_installments'], axis = 1, inplace = True)
    max_due_install_min_period['min_period'] = pandas.to_numeric(max_due_install_min_period['min_period'])
    return max_due_install_min_period

def DefaultPerProduct(data_product, n_installments_due = 3, max_period = 200712):
    #Creating dataframe with 0-1 variable if the product defaulted
    import pandas
    col_name = 'default'+str(n_installments_due)
    data_product.insert(len(data_product.columns),col_name, 0)
    for index, row in data_product.iterrows():        
        if row['max_due_installments'] >= n_installments_due and row['min_period'] <= max_period:
            data_product.at[index, col_name] = 1
        else:
            data_product.at[index, col_name] = 0
    data_product.drop(['min_period', 'max_due_installments'], axis = 1, inplace = True)
    return data_product

def ProductionDefault(data_product_default, data_production):
    import pandas
    #Adding col1umn with 0-1 default to the main dataframe
    data_production['period'] = pd.to_numeric(data_production['period'])
    data_production = data_production[data_production['period'] <= 200712]
    data_merged = data_production.merge(right = data_product_default, on = 'aid')
    return data_merged

