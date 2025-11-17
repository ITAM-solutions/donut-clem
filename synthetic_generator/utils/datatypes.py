"""
File name: datatypes
Author: Fran Moreno
Last Updated: 9/1/2025
Version: 1.0
Description: TOFILL
"""

from enum import Enum
from dataclasses import dataclass


class FieldType(str, Enum):
    Currency = 'currency'
    Date = 'date'
    InvoiceID = 'idInvoice'
    PONumber = 'poNumber'
    IDLong = 'idLong'
    IDShort = 'idShort'
    IDHuge = 'idHuge'
    IDTax = 'idTax',
    IDBank = "idBank"
    IDNumeric = "idNumeric"
    Price = 'price'
    Text = 'text'
    ProductName = 'productName'
    ProductDesc = 'productDesc'
    ProductDescShort = 'productDescShort'
    ShortText = 'shortText'
    LongText = 'longText'
    CompanyName = 'companyName'
    Number = 'number'
    Metric = 'metric'
    MetricGroup = 'metricGroup'
    BillingTerm = 'billingTerm'
    PaymentTerm = 'paymentTerm'
    DateRange = 'dateRange'
    Percentage = 'percentage'
    DocType = 'docType'
    PhoneNumber = 'phoneNumber'
    Email = 'email',
    PersonName = 'name',
    Signature = 'signature',
    WorkTitle = 'workTitle'
    Address = 'address',
    BankName = 'bankName',
    IBAN = 'iban',
    Swift = 'swift',
    PaymentMethod = 'paymentMethod'
    Index = 'index'
    UUID = 'uuid'
    Street = 'street'
    City = 'city'
    Country = 'country'
    PostalCode = 'postalCode'
    Url = 'url'
