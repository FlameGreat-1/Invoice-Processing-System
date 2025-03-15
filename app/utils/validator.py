from typing import List, Dict, Tuple
from datetime import datetime, date
from decimal import Decimal
from app.models import Invoice, Vendor, Address, InvoiceItem
from app.config import settings
import re
import logging
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class InvoiceValidator:
    def __init__(self):
        # We don't have a predefined list of approved vendors
        pass

    def validate_invoice(self, invoice: Invoice) -> Tuple[bool, List[str]]:
        errors = []

        # Validate invoice number
        if not self._validate_invoice_number(invoice.invoice_number):
            errors.append(f"Invalid invoice number format: {invoice.invoice_number}")

        # Validate vendor information
        vendor_errors = self._validate_vendor(invoice.vendor)
        errors.extend(vendor_errors)

        # Validate date
        date_errors = self._validate_date(invoice.invoice_date)
        errors.extend(date_errors)

        # Validate totals
        if not self._validate_totals(invoice.grand_total, invoice.taxes, invoice.final_total):
            errors.append("Total amounts do not match: grand_total + taxes != final_total")

        # Validate multi-page consistency
        if not self._validate_multi_page(invoice):
            errors.append("Inconsistent multi-page information")

        return len(errors) == 0, errors

    def _validate_invoice_number(self, invoice_number: str) -> bool:
            # Check if the invoice number is not empty and contains at least one alphanumeric character
        return bool(invoice_number) and any(char.isalnum() for char in invoice_number)

    def _validate_vendor(self, vendor: Vendor) -> List[str]:
        errors = []
        if not vendor.name or not vendor.name.strip():
            errors.append("Vendor name is missing")
        if not vendor.address or not vendor.address.street or not vendor.address.city:
            errors.append("Vendor address is incomplete")
        return errors

    def _validate_date(self, invoice_date: date) -> List[str]:
        errors = []
        if invoice_date > date.today():
            errors.append(f"Invoice date {invoice_date} is in the future")
        return errors

    def _validate_totals(self, grand_total: Decimal, taxes: Decimal, final_total: Decimal) -> bool:
        # Ensure totals match exactly (Grand + Tax = Final)
        return (grand_total + taxes) == final_total

    def _validate_multi_page(self, invoice: Invoice) -> bool:
    # Check if the number of pages is valid (1 or more)
    return invoice.pages >= 1

    def validate_extracted_data(self, extracted_data: Dict) -> Tuple[bool, List[str]]:
        try:
            invoice = Invoice(**extracted_data)
            return self.validate_invoice(invoice)
        except ValidationError as e:
            return False, [str(e)]

invoice_validator = InvoiceValidator()

def validate_invoice_batch(invoices: List[Dict]) -> List[Tuple[Dict, bool, List[str]]]:
    results = []
    for invoice_data in invoices:
        is_valid, errors = invoice_validator.validate_extracted_data(invoice_data)
        results.append((invoice_data, is_valid, errors))
    return results

def flag_anomalies(invoices: List[Invoice]) -> List[Dict]:
    flagged_invoices = []
    for invoice in invoices:
        flags = []
        
        # Flag future dates
        if invoice.invoice_date > date.today():
            flags.append("Future date")

        # Flag unusually high amounts (threshold can be adjusted)
        if invoice.final_total > Decimal('10000.00'):
            flags.append("Unusually high total amount")

        # Flag invoices with many line items
        if len(invoice.items) > 20:
            flags.append("Large number of line items")

        if flags:
            flagged_invoices.append({**invoice.dict(), 'flags': flags})

    return flagged_invoices
