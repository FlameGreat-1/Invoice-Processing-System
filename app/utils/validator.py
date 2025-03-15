from typing import List, Dict, Tuple
from datetime import datetime, date
from decimal import Decimal
from app.models import Invoice, Vendor
from app.config import settings
import re
import logging
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class InvoiceValidator:
    def __init__(self):
        self.approved_vendors = self._load_approved_vendors()

    def _load_approved_vendors(self) -> Dict[str, Vendor]:
        # In a real-world scenario, this would load from a database or external service
        # For this implementation, we'll use a mock list
        return {
            "Acme Corp": Vendor(name="Acme Corp", address={"street": "123 Main St", "city": "Anytown", "state": "CA", "country": "US", "postal_code": "12345"}),
            "Global Services Inc.": Vendor(name="Global Services Inc.", address={"street": "456 Oak Ave", "city": "Somewhere", "state": "NY", "country": "US", "postal_code": "67890"}),
        }

    def validate_invoice(self, invoice: Invoice) -> Tuple[bool, List[str]]:
        errors = []

        # Validate invoice number
        if not self._validate_invoice_number(invoice.invoice_number):
            errors.append(f"Invalid invoice number format: {invoice.invoice_number}")

        # Validate vendor
        if not self._validate_vendor(invoice.vendor):
            errors.append(f"Unapproved vendor: {invoice.vendor.name}")

        # Validate date
        if not self._validate_date(invoice.invoice_date):
            errors.append(f"Invalid invoice date: {invoice.invoice_date}")

        # Validate totals
        if not self._validate_totals(invoice.grand_total, invoice.taxes, invoice.final_total):
            errors.append("Total amounts do not match: grand_total + taxes != final_total")

        # Validate line items
        item_errors = self._validate_line_items(invoice.items)
        errors.extend(item_errors)

        return len(errors) == 0, errors

    def _validate_invoice_number(self, invoice_number: str) -> bool:
        # Implement a strict invoice number format check
        pattern = r'^INV-\d{3}$'
        return re.match(pattern, invoice_number) is not None

    def _validate_vendor(self, vendor: Vendor) -> bool:
        return vendor.name in self.approved_vendors

    def _validate_date(self, invoice_date: date) -> bool:
        # Check if the invoice date is not in the future
        return invoice_date <= date.today()

    def _validate_totals(self, grand_total: Decimal, taxes: Decimal, final_total: Decimal) -> bool:
        # Ensure totals match with a small tolerance for floating-point arithmetic
        return abs((grand_total + taxes) - final_total) < Decimal('0.01')

    def _validate_line_items(self, items: List[Dict]) -> List[str]:
        errors = []
        for idx, item in enumerate(items, start=1):
            try:
                # Validate quantity
                if item['quantity'] <= 0:
                    errors.append(f"Invalid quantity for item {idx}: {item['quantity']}")

                # Validate unit price
                if item['unit_price'] <= Decimal('0.00'):
                    errors.append(f"Invalid unit price for item {idx}: {item['unit_price']}")

                # Validate total
                expected_total = item['quantity'] * item['unit_price']
                if abs(item['total'] - expected_total) > Decimal('0.01'):
                    errors.append(f"Total mismatch for item {idx}: expected {expected_total}, got {item['total']}")

            except KeyError as e:
                errors.append(f"Missing required field in item {idx}: {str(e)}")
            except (TypeError, ValueError) as e:
                errors.append(f"Invalid data type in item {idx}: {str(e)}")

        return errors

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

def flag_anomalies(invoices: List[Dict]) -> List[Dict]:
    flagged_invoices = []
    for invoice in invoices:
        flags = []
        
        # Flag unusually high amounts
        if invoice['final_total'] > Decimal('10000.00'):
            flags.append("Unusually high total amount")

        # Flag invoices with many line items
        if len(invoice['items']) > 20:
            flags.append("Large number of line items")

        # Flag invoices with very low or very high tax rates
        if invoice['taxes'] > Decimal('0.00'):
            tax_rate = invoice['taxes'] / invoice['grand_total']
            if tax_rate < Decimal('0.05') or tax_rate > Decimal('0.25'):
                flags.append("Unusual tax rate")

        if flags:
            flagged_invoices.append({**invoice, 'flags': flags})

    return flagged_invoices
    