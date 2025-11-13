"""
File name: word_bank
Author: Fran Moreno
Last Updated: 8/28/2025
Version: 1.0
Description: TOFILL
"""
from enum import Enum
from pathlib import Path

from synthetic_generator.utils.datatypes import FieldType

SYNTH_GEN_PATH = Path(r'.')

""" Data corpus """

with open(SYNTH_GEN_PATH / "data" / "corpus.txt", "r") as f:
    WORDS = [word.replace('\n', '') for word in f.readlines()]

with open(SYNTH_GEN_PATH / "data" / "publishers.txt", "r", encoding="utf-8") as f:
    COMPANY_NAMES = [word.replace('\n', '') for word in f.readlines()]

with open(SYNTH_GEN_PATH / "data" / "products.txt", "r", encoding="utf-8") as f:
    PRODUCT_NAMES = [word.replace('\n', '') for word in f.readlines()]


""" FORMAT: [possible_values] """

DOCUMENT_TYPE = (
    "INVOICE",
    "Invoice",
    "Quotation",
    "Renewal",
    "Quote",
    "Suplemental Quote",
    "Software License",
    "Contract",
    "Software Contract",
    "License",
    "Subscription",
    "Software Subscription",
)

PAYMENT_TERMS = [
    "Due on Receipt",
    "Immediate",
    "Cash on Delivery",
    "Net 7",
    "Net 10",
    "Net 15",
    "Net 30",
    "Net 45",
    "Net 60",
    "Net 90",
    "Net 30 EOM",
    "EOM 15",
    "Prepaid",
    "Advance Payment",
    "Upfront",
    "Quarterly payments",
    "2/10 Net 30",
]

BILLING_TERMS = [
    "Single Billing",
    "One-off",
    "Per Month",
    "Every 3 Months",
    "Quarterly Billing",
    "Half-Yearly",
    "Semi-Annual",
    "Annual Billing",
    "Per Year",
    "12 Months",
    "Weekly",
    "Daily",
    "Per Project",
    "Per Usage",
    "Pay-As-You-Go"
]

METRICS = [
    "Per User",
    "Per Named User",
    "Per Concurrent User",
    "Per Active User",
    "Per Device/User",
    "Per Device",
    "Per Workstation",
    "Per Server",
    "Per CPU",
    "Per Core",
    "Per Socket",
    "Installation",
    "Processor",
    "Processor Core",
    "PVU",
    "Total Users",
    "Concurrent Users",
    "CAL User",
    "CAL Device",
    "Total Devices",
    "Concurrent Devices",
    "Named User",
    "CAL",
    "Devices",
    "Total cores",
    "Total processors",
    "Users",
    "Custom Compare Value"
    "PVU",
    "User Subscription"
]

METRIC_GROUPS = [
    "Subscription",
    "Installations",
    "Number of processor cores",
    "Number of Processors",
    "CAL",
    "Common",
    "Concurrent Licenses",
    "Consumption",
    "SAP",
]

BANK_PAYMENT_METHODS = [
    "by Check",
    "by Transference",
    "Electronic Funds Transfer",
]

BANK_NAMES = [
    # Netherlands
    "ING Bank",
    "Rabobank",
    "ABN AMRO Bank",
    "BNG Bank",
    "Nederlandse Waterschapsbank",
    "ASN Bank",
    "De Volksbank",
    "NIBC Bank",
    "Van Lanschot Kempen",
    "Achmea Bank",
    "Triodos Bank",

    # Europe (major groups)
    "HSBC",
    "Barclays",
    "Lloyds Banking Group",
    "NatWest Group",
    "BNP Paribas",
    "Crédit Agricole Group",
    "Société Générale",
    "Crédit Mutuel Group",
    "Banco Santander",
    "BBVA",
    "Deutsche Bank",
    "Commerzbank",
    "UBS Group",
    "Credit Suisse",
    "Intesa Sanpaolo",
    "UniCredit",
    "Nordea",
    "Danske Bank",
    "KBC Group",
    "Erste Group Bank",
    "Raiffeisen Bank International",
    "SEB",
    "Swedbank",
    "Handelsbanken"
]

DOCUMENT_SECTIONS = [
    "Products / Services",
    "Bundles / Packages",
    "Service Levels / SLAs",
    "Discounts",
    "Credits / Adjustments",
    "Renewal Terms",
    "Subtotal / Totals",
    "Currency Details",
    "Exchange Rates",
    "Installment Schedules",
    "Payment Due Dates",
    "Late Payment Penalties / Interest",
    "Banking Details / Wire Instructions",
    "Supported Payment Methods",
    "Terms & Conditions",
    "Usage Rights & License Grants",
    "Restrictions & Limitations",
    "Confidentiality / NDA Clauses",
    "Termination / Cancellation Policy",
    "Warranty & Liability Clauses",
    "Force Majeure",
    "Dispute Resolution / Governing Law",
    "Audit Rights",
    "Data Protection & GDPR Clauses",
    "Customer Details",
    "Vendor / Supplier Details",
    "Purchase Order (PO) References",
    "Contract Reference Numbers",
    "Tax Identification Numbers",
    "Tax Breakdown",
    "Reverse Charge / Self-Assessment Notes",
    "Compliance Certifications",
    "Export / Import Restrictions",
    "Delivery Instructions",
    "Digital Access / Activation Keys",
    "License File Delivery",
    "Support Contacts & Escalation Paths",
    "Implementation / Onboarding Plans",
    "Notes / Memos",
    "Payment Tips",
    "Customer Self-Service Links / Portals",
    "FAQs / Helpdesk Links",
    "Acknowledgment / Acceptance Signatures",
    "Surcharges / Fees",
    "Service Extensions / Add-ons",
    "Trial-to-Paid Upgrade Options",
    "Future Renewal Reminders",
]

CURRENCY_TYPE = [
    'EUR',
    'USD',
    'EURO',
    'DOLLAR',
    '$',
    '€'
]

""" GENERAL CONFIGURATION """

DATE_FORMATS = [
    '%Y-%m-%d',
    '%d-%m-%Y',
    '%m-%d-%Y',
    '%Y/%m/%d',
    '%d/%m/%Y',
    '%m/%d/%Y',
    '%Y.%m.%d',
    '%d.%m.%Y',
    '%m.%d.%Y',
    '%d %B %Y',
    '%B %d, %Y',
    '%Y %B %d',
    '%B %Y',
    '%d %b %Y',
    '%b %d, %Y',
    '%Y %b %d',
    '%b %Y',
]

DATE_RANGE_SEPARATORS = [
    '-',
    'to',
    'till',
    '|',
    '~'
]


FONT_FAMILIES = (
    "Arial",
    "Calibri",
    "Times New Roman",
    "Verdana",
    "Tahoma",
    "Trebuchet MS",
    "Georgia",
    "Helvetica",
    "Cambria",
    "Century Gothic",
    "Segoe UI",
)

FONT_SIZES = (
    12,
    18,
    20,
    22,
    24,
    26,
    28,
)

TABLE_FONT_SIZES = {
    2: 24,   # very spacious -> normal invoice text
    3: 22,
    4: 22,
    5: 19,
    6: 18,
    7: 17,
    8: 16,
    9: 15,
    10: 14,
    11: 13,
    12: 12,   # tight but still legible
    13: 11,
    14: 10,
}


FONT_COLORS = (
    '#000000',
    '#212529',
    '#1C1C1C',
    '#2E2E2E',
    '#1E1E1E',
    '#202020',
)

PAGE_IDX_FORMATS = (
    'Page #x of #y',
    '#x-#y',
    '#x',
    '#x/#y',
    '#x (of #y)',
    '#x of #y pages',
    'Page #x',
    'Page n.#x'
)

""" FORMAT: id: (DataType, [repr_values] """

INVOICE_MAIN_FIELDS = {
    'id': (FieldType.InvoiceID, ('Invoice ID', 'Invoice Number', 'Invoice #', 'Invoice No.', 'Inv. No.', 'Document Number', 'Bill Number', 'Reference Number', 'Ref. No.', 'Quotation ID', 'Quotation #', 'Quotation Number')),
    'date': (FieldType.Date, ('Invoice Date', 'Date', 'Inv. Date', 'Document Date', 'Billing Date', 'Issue Date')),
    'po': (FieldType.PONumber, ('P.O. Number', 'PO Number', 'Purchase Order Number', 'Order Reference', 'PO Ref.', 'Purchase Ref.', 'Purchase no.')),
    'cur': (FieldType.Currency, ('Currency', 'Curr.', 'Invoice Currency', 'Payment Currency', 'Currency Code')),
    'vendor': (FieldType.CompanyName, ('Vendor Name', 'Provider', 'Service Provider', 'Seller', 'Contractor', 'From', 'Inv. From')),
    'company': (FieldType.CompanyName, ('Customer', 'Owner', 'Purchaser', 'Recipient', 'End User', 'Organization', 'Partner', 'Beneficiary', 'Invoiced Party', 'Billed Party', 'Client', 'To'))
}

INVOICE_ADDITIONAL_FIELDS = {
    'billing_terms': (FieldType.BillingTerm, ('Billing Terms', 'Terms of Billing', 'Invoicing Terms', 'Invoice Terms', 'Contract Terms')),
    'payment_terms': (FieldType.PaymentTerm, ('Payment Terms', 'Terms of Payment', 'Payment Conditions', 'Payment Agreement', 'Senttlement Terms')),
    'due_date': (FieldType.Date, ('Due Date', 'Payment Due Date', 'Invoice Due Date', 'Pay By Date', 'Deadline', 'Payment Deadline')),
    'billing_period': (FieldType.DateRange, ('Billing Period', 'Invoice Period', 'Period of Performance', 'Service Period', 'Charge Period')),
    'billing_account': (FieldType.IDBank, ('Billing Account', 'Account Number', 'Customer Account', 'Invoice Account', 'Payment Account')),
    'customer_tax': (FieldType.IBAN, ('Customer Tax', 'Tax ID', 'VAT Number', 'VAT ID', 'GST Number', 'Business Tax Number')),
    'doc_type': (FieldType.DocType, ('Document Type', 'Type of Document', 'Document', 'File', 'Type')),
    'subtotal': (FieldType.Price, ('Subtotal', 'Sub Total', 'Total (excl. VAT)')),
    'total': (FieldType.Price, ('Total', 'Total Amount', 'Total (incl. VAT)')),
    'tax_code': (FieldType.IDShort, ('Tax Code', 'Tax ID')),
    'tax_amount': (FieldType.Price, ('Tax Amount', 'Tax Total')),
    'cust_id': (FieldType.IDNumeric, ('Customer ID', 'Cust. ID', 'Customer number', 'Customer', 'Cust.')),
    'vendor_id': (FieldType.IDNumeric, ('Vendor No.', 'Vendor Number', 'Vendor ID', 'Seller ID')),
    'contract_id': (FieldType.IDNumeric, ('Contract ID', 'Contract Number', 'Contract #', 'Contract Identifier', 'Agreement Number', 'Agreement #'))
}

PRODUCT_SHARED_FIELDS = {
    'qty': (FieldType.Number, ('Quantity', 'Qty', 'Amount', 'Units', 'No. of Items', 'Volume')),
    'dfrom': (FieldType.Date, ('Date From', 'From', 'Start Date', 'Service From', 'Subscription start', 'Period Start')),
    'dto': (FieldType.Date, ('Date To', 'To', 'End Date', 'Service To', 'Subscription end', 'Period End')),
    'met': (FieldType.Metric, ('Metric', 'Unit', 'Measurement Unit', 'UOM', 'Unit of Measure', 'Measure')),
    'metgr': (FieldType.MetricGroup, ('Metric Group', 'Unit Group', 'Category', 'Measurement Category', 'Grouping')),
}

PRODUCT_MAIN_FIELDS = {
    'sku': (FieldType.IDShort, ('SKU', 'Stock Keeping Unit', 'Product Code', 'Code', 'Item Code', 'Identifier', 'Article Number', 'Part #')),
    'name': (FieldType.ProductName, ('Product Name', 'Item Name', 'Name', 'Service Name', 'Service', 'Item')),
    'qty': (FieldType.Number, ('Quantity', 'Qty', 'Amount', 'Units', 'No. of Items', 'Volume')),
    'unpr': (FieldType.Price, ('Unit Price', 'Price per Unit', 'Rate', 'Unit Cost', 'Price Each', 'Cost per Item')),
    'totpr': (FieldType.Price, ('Total Price', 'Total', 'Line Total', 'Amount', 'Subtotal')),
    'dfrom': (FieldType.Date, ('Date From', 'From', 'Start Date', 'Service From', 'Subscription start', 'Period Start')),
    'dto': (FieldType.Date, ('Date To', 'To', 'End Date', 'Service To', 'Subscription end', 'Period End')),
    'met': (FieldType.Metric, ('Metric', 'Unit', 'Measurement Unit', 'UOM', 'Unit of Measure', 'Measure')),
    'metgr': (FieldType.MetricGroup, ('Metric Group', 'Unit Group', 'Category', 'Measurement Category', 'Grouping')),
}

PRODUCT_ADDITIONAL_FIELDS = {
    'idx': (FieldType.Index, ('Index', 'Idx.', 'Number', 'ID', 'Item')),
    'disc_rate': (FieldType.Percentage, ('Discount', 'Rebate', 'Deduction', 'Price Deduction', 'Special Offer')),
    'disc_amount': (FieldType.Price, ('Discount Amount', 'Discount', 'Deduction', 'Discounted')),
    'tax_rate': (FieldType.Percentage, ('Tax Rate', 'VAT %', 'GST %', 'Tax %', 'Tax Percentage')),
    'tax_amount': (FieldType.Price, ('Sales Tax', 'VAT', 'GST', 'Tax', 'Tax Amount')),
    "subscription_period": (FieldType.DateRange, ("Subscription Period", "Period", "Service Period", "Billing Period", "Coverage Period", "Contract Period", "Duration")),
    "serial": (FieldType.UUID, ("Serial Number", "Serial No.", "Product Serial")),
    "desc": (FieldType.ProductDesc, ("Description", "Product Description", "Item Description")),
    "desc_short": (FieldType.ProductDescShort, ("Description", "Summary")),
    "deal_type": (FieldType.DocType, ("Contract Type", "Type")),
}

PRODUCT_FIELD_GROUPS = [
    ("qty", "name", "unpr", "totpr", "disc_rate"),
    ("name", "totpr"),
    ("name", "qty", "unpr", "totpr"),
    ("name", "sku", "dfrom", "dto", "qty", "unpr", "totpr"),
    ("name", "sku", "qty", "disc_rate", "unpr", "totpr"),
    ("name", "qty", "unpr", "disc_rate", "totpr"),
    ("sku", "name", "qty", "unpr", "totpr"),
    ("sku", "name", "dfrom", "dto", "qty", "totpr"),
    ("qty", "sku", "name", "met", "metgr", "dfrom", "dto", "unpr", "totpr"),
    ("sku", "name", "dfrom", "dto", "met", "metgr", "unpr", "disc_rate", "totpr"),
    ("name", "metgr", "dfrom", "dto", "sku", "totpr"),
    ("name", "sku"),
    ("subscription_period", "name", "met", "unpr", "totpr"),
    ("idx", "name", "totpr"),
    ("name", "unpr", "tax_rate", "tax_amount", "disc_rate", "totpr"),
]

PRODUCT_FIELD_GROUPS_SPECIAL = {
    "017": ("idx", "name", "qty", "met", "unpr", "totpr", "tax_rate", "tax_amount", "disc_rate", "disc_amount"),
    "020": ("qty", "name", "sku", "subscription_period", "unpr", "totpr"),
    "021": ("idx", "sku", "name", "qty", "dfrom", "dto", "tax_amount", "disc_amount", "unpr", "totpr", "serial", "tax_rate"),
    "022": ("sku", "name", "qty", "dfrom", "dto", "desc", "desc_short", "deal_type", "tax_rate", "met"),
}

CONTACT_INFO_FIELDS = {
    'name': (FieldType.PersonName, ('Name', 'Authorizer', 'Owner', 'Support', 'Contact')),
    'number': (FieldType.PhoneNumber, ('Contact Number', 'Phone Number', 'Number', )),
    'email': (FieldType.Email, ('Email Address', 'Address', 'Email', )),
}

SIGNATURE_INFO_FIELDS = {
    'vendor_name': (FieldType.PersonName, ('Authorizer', 'Name')),
    'vendor_signature': (FieldType.Signature, ('Signature', 'Signed by', 'Signed')),
    'vendor_title': (FieldType.WorkTitle, ('Title', 'Role', 'Job')),
    'customer_name': (FieldType.PersonName, ('Name', )),
    'customer_signature': (FieldType.Signature, ('Signature', 'Signed By', 'Signed')),
    'date': (FieldType.Date, ('Date', 'Signature Date')),
    'location': (FieldType.Address, ('Signed in', 'Place', 'Location', 'Signature Address'))
}

BANK_INFO_FIELDS = {
    'name': (FieldType.BankName, ('Bank Name', 'Bank')),
    'id': (FieldType.IDNumeric, ('Bank ID', 'ID')),
    'iban': (FieldType.IBAN, ('IBAN', 'IBAN Number', 'Account Number', 'Account')),
    'swift': (FieldType.Swift, ('SWIFT Code', 'SWIFT')),
    'payment_method': (FieldType.PaymentMethod, ('To Pay', 'Payment Method')),
    'acc_name': (FieldType.CompanyName, ('Account Name', 'Beneficiary', 'Account', '')),
    'address': (FieldType.Address, ('Bank Address', 'Address')),
}


DOCUMENT_IDS = {
    'uuid': (FieldType.UUID, ('UUID', 'GUID', 'Invoice UUID', 'Identifier')),
    'hash': (FieldType.IDLong, ('Document Hash', 'Hash Code', 'SHA256', 'MD5', 'Document Signature')),
    'transaction': (FieldType.IDHuge, ('TxID', 'Ref. ID', 'Reference ID', 'Operation ID')),
    'signature': (FieldType.UUID, ('Signature ID', 'Sign. ID', 'SigCode', 'eSign ID', 'ID')),
    'license': (FieldType.IDLong, ('License', 'License Key', 'License Number', 'License ID', 'Activation Key')),
}

""" FORMAT: id: [possible_values]"""



""" (value,  probability) """

LANGUAGES = (
    ('en_US', 1.0),
    # ('nl_NL', 0.3)
)

COLOR_PALETTES = [
    (('#FFFFFF', '#FFFFFF', '#000000', '#A1A1A1'), 0.40),      # classic, most common
    (('#FFFFFF', '#000000', '#000000', '#A1A1A1'), 0.04),      # classic, most common
    (('#F8F9FA', '#6C757D', '#212529', '#A1A1A1'), 0.04),      # soft_gray
    (('#F4F7FB', '#4A6FA5', '#1C1C1C', '#4FACE1'), 0.04),      # blue_tint
    (('#FAF6F1', '#A68A64', '#2E2E2E', '#E2C598'), 0.04),      # warm_beige
    (('#F5FAF7', '#4F6D5B', '#1E1E1E', '#84AE9B'), 0.04),      # cool_green
    (('#F9FAFB', '#5A5A5A', '#202020', '#9C9C9C'), 0.04),      # neutral_slate
    (('#FFFFFF', '#007BFF', '#000000', '#39acff'), 0.04),      # white + primary blue
    (('#F2F2F2', '#343A40', '#212529', '#54656C'), 0.04),      # light gray + dark gray
    (('#FFF8E7', '#FFC107', '#212529', '#FFF089'), 0.04),      # soft yellow + amber
    (('#EFFAF9', '#17A2B8', '#1C1C1C', '#17D6EC'), 0.04),      # pale cyan + teal
    (('#F0F5FF', '#6610F2', '#1C1C1C', '#AC80F2'), 0.04),      # pale violet + purple
    (('#FAFCF5', '#28A745', '#1E1E1E', '#6CA77E'), 0.04),      # very light green + green
    (('#FCF9F7', '#6F42C1', '#202020', '#907CC1'), 0.04),      # soft cream + deep purple
    (('#FDF5F0', '#DC3545', '#2E2E2E', '#DC838E'), 0.04),      # soft pink + red
    (('#F7F9F9', '#20C997', '#1C1C1C', '#85C9AF'), 0.04),      # pale mint + green
]


""" ENUMS """

class PriceFormat(Enum):
    NO_TH_DOT_DEC = 0
    NO_TH_COMMA_DEC = 1
    COMMA_TH_DOT_DEC = 2
    DOT_TH_COMMA_DEC = 3
    SPACE_TH_DOT_DEC = 4


class NumberFormat(Enum):
    NO_TH = 0
    DOT_TH = 0
    COMMA_TH = 0
