"""
Cost tracking module for MLOps SageMaker Demo.

This module provides functions for tracking and analyzing AWS costs
using the AWS Cost Explorer API.
"""

import boto3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging

# Configure logging
logger = logging.getLogger(__name__)

class CostTracker:
    """Cost tracking class for AWS resources."""
    
    def __init__(self, session=None, profile_name='ab'):
        """
        Initialize the CostTracker.
        
        Args:
            session: Boto3 session to use. If None, a new session will be created.
            profile_name: AWS profile name to use if session is None.
        """
        if session is None:
            self.session = boto3.Session(profile_name=profile_name)
        else:
            self.session = session
        
        self.ce_client = self.session.client('ce')
        self.project_name = 'mlops-sagemaker-demo'
    
    def get_cost_and_usage(
        self,
        start_date: str,
        end_date: str,
        granularity: str = 'DAILY',
        metrics: List[str] = ['UnblendedCost'],
        group_by: Optional[List[Dict]] = None,
        filter_expression: Optional[Dict] = None
    ) -> Dict:
        """
        Get cost and usage data from AWS Cost Explorer.
        
        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            granularity: Time granularity (DAILY, MONTHLY, HOURLY).
            metrics: List of metrics to retrieve.
            group_by: List of dimensions to group by.
            filter_expression: Filter expression for the query.
            
        Returns:
            Dict containing cost and usage data.
        """
        try:
            kwargs = {
                'TimePeriod': {
                    'Start': start_date,
                    'End': end_date
                },
                'Granularity': granularity,
                'Metrics': metrics
            }
            
            if group_by:
                kwargs['GroupBy'] = group_by
            
            if filter_expression:
                kwargs['Filter'] = filter_expression
            
            response = self.ce_client.get_cost_and_usage(**kwargs)
            return response
        except Exception as e:
            logger.error(f"Error getting cost and usage data: {e}")
            return {'ResultsByTime': []}
    
    def get_project_costs(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        granularity: str = 'DAILY'
    ) -> Dict:
        """
        Get costs for the project.
        
        Args:
            start_date: Start date in YYYY-MM-DD format. If None, defaults to 30 days ago.
            end_date: End date in YYYY-MM-DD format. If None, defaults to today.
            granularity: Time granularity (DAILY, MONTHLY, HOURLY).
            
        Returns:
            Dict containing cost data for the project.
        """
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Create filter for project tag
        filter_expression = {
            'Tags': {
                'Key': 'Project',
                'Values': [self.project_name]
            }
        }
        
        # Group by service
        group_by = [
            {
                'Type': 'DIMENSION',
                'Key': 'SERVICE'
            }
        ]
        
        return self.get_cost_and_usage(
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
            metrics=['UnblendedCost'],
            group_by=group_by,
            filter_expression=filter_expression
        )
    
    def get_sagemaker_costs_by_resource(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        granularity: str = 'DAILY'
    ) -> Dict:
        """
        Get SageMaker costs grouped by resource.
        
        Args:
            start_date: Start date in YYYY-MM-DD format. If None, defaults to 30 days ago.
            end_date: End date in YYYY-MM-DD format. If None, defaults to today.
            granularity: Time granularity (DAILY, MONTHLY, HOURLY).
            
        Returns:
            Dict containing SageMaker cost data grouped by resource.
        """
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Create filter for SageMaker service and project tag
        filter_expression = {
            'And': [
                {
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon SageMaker']
                    }
                },
                {
                    'Tags': {
                        'Key': 'Project',
                        'Values': [self.project_name]
                    }
                }
            ]
        }
        
        # Group by resource
        group_by = [
            {
                'Type': 'DIMENSION',
                'Key': 'RESOURCE'
            }
        ]
        
        return self.get_cost_and_usage(
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
            metrics=['UnblendedCost'],
            group_by=group_by,
            filter_expression=filter_expression
        )
    
    def get_costs_by_tag(
        self,
        tag_key: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        granularity: str = 'DAILY'
    ) -> Dict:
        """
        Get costs grouped by a specific tag.
        
        Args:
            tag_key: Tag key to group by.
            start_date: Start date in YYYY-MM-DD format. If None, defaults to 30 days ago.
            end_date: End date in YYYY-MM-DD format. If None, defaults to today.
            granularity: Time granularity (DAILY, MONTHLY, HOURLY).
            
        Returns:
            Dict containing cost data grouped by the specified tag.
        """
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Create filter for project tag
        filter_expression = {
            'Tags': {
                'Key': 'Project',
                'Values': [self.project_name]
            }
        }
        
        # Group by the specified tag
        group_by = [
            {
                'Type': 'TAG',
                'Key': tag_key
            }
        ]
        
        return self.get_cost_and_usage(
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
            metrics=['UnblendedCost'],
            group_by=group_by,
            filter_expression=filter_expression
        )
    
    def get_cost_forecast(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        granularity: str = 'MONTHLY',
        prediction_interval_level: int = 80
    ) -> Dict:
        """
        Get cost forecast from AWS Cost Explorer.
        
        Args:
            start_date: Start date in YYYY-MM-DD format. If None, defaults to today.
            end_date: End date in YYYY-MM-DD format. If None, defaults to 30 days from now.
            granularity: Time granularity (DAILY, MONTHLY).
            prediction_interval_level: Prediction interval level (70-99).
            
        Returns:
            Dict containing cost forecast data.
        """
        # Set default dates if not provided
        if not start_date:
            start_date = datetime.now().strftime('%Y-%m-%d')
        
        if not end_date:
            end_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Create filter for project tag
        filter_expression = {
            'Tags': {
                'Key': 'Project',
                'Values': [self.project_name]
            }
        }
        
        try:
            response = self.ce_client.get_cost_forecast(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Metric='UNBLENDED_COST',
                Granularity=granularity,
                Filter=filter_expression,
                PredictionIntervalLevel=prediction_interval_level
            )
            return response
        except Exception as e:
            logger.error(f"Error getting cost forecast: {e}")
            return {'ForecastResultsByTime': []}
    
    def get_cost_anomalies(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        anomaly_monitor: Optional[str] = None
    ) -> Dict:
        """
        Get cost anomalies from AWS Cost Explorer.
        
        Args:
            start_date: Start date in YYYY-MM-DD format. If None, defaults to 30 days ago.
            end_date: End date in YYYY-MM-DD format. If None, defaults to today.
            anomaly_monitor: Anomaly monitor ID. If None, all monitors are used.
            
        Returns:
            Dict containing cost anomaly data.
        """
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        try:
            kwargs = {
                'DateInterval': {
                    'StartDate': start_date,
                    'EndDate': end_date
                }
            }
            
            if anomaly_monitor:
                kwargs['MonitorArn'] = anomaly_monitor
            
            response = self.ce_client.get_anomalies(**kwargs)
            return response
        except Exception as e:
            logger.error(f"Error getting cost anomalies: {e}")
            return {'Anomalies': []}
    
    def create_cost_budget(
        self,
        budget_name: str,
        budget_amount: float,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        notification_email: Optional[str] = None,
        threshold_percent: float = 80.0
    ) -> Dict:
        """
        Create a cost budget for the project.
        
        Args:
            budget_name: Name of the budget.
            budget_amount: Budget amount in USD.
            start_date: Start date in YYYY-MM-DD format. If None, defaults to first day of current month.
            end_date: End date in YYYY-MM-DD format. If None, defaults to last day of current month.
            notification_email: Email address for notifications. If None, no notifications are set up.
            threshold_percent: Threshold percentage for notifications.
            
        Returns:
            Dict containing the budget creation response.
        """
        # Set default dates if not provided
        if not start_date:
            today = datetime.now()
            start_date = datetime(today.year, today.month, 1).strftime('%Y-%m-%d')
        
        if not end_date:
            today = datetime.now()
            if today.month == 12:
                end_date = datetime(today.year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(today.year, today.month + 1, 1) - timedelta(days=1)
            end_date = end_date.strftime('%Y-%m-%d')
        
        # Create filter for project tag
        filter_expression = {
            'Tags': {
                'Key': 'Project',
                'Values': [self.project_name]
            }
        }
        
        try:
            budgets_client = self.session.client('budgets')
            
            budget_config = {
                'BudgetName': budget_name,
                'BudgetType': 'COST',
                'BudgetLimit': {
                    'Amount': str(budget_amount),
                    'Unit': 'USD'
                },
                'CostFilters': {},
                'CostTypes': {
                    'IncludeTax': True,
                    'IncludeSubscription': True,
                    'UseBlended': False,
                    'IncludeRefund': False,
                    'IncludeCredit': False,
                    'IncludeUpfront': True,
                    'IncludeRecurring': True,
                    'IncludeOtherSubscription': True,
                    'IncludeSupport': True,
                    'IncludeDiscount': True,
                    'UseAmortized': False
                },
                'TimeUnit': 'MONTHLY',
                'TimePeriod': {
                    'Start': datetime.strptime(start_date, '%Y-%m-%d'),
                    'End': datetime.strptime(end_date, '%Y-%m-%d')
                }
            }
            
            response = budgets_client.create_budget(
                AccountId=self.session.client('sts').get_caller_identity()['Account'],
                Budget=budget_config
            )
            
            # Create notification if email is provided
            if notification_email:
                notification = {
                    'NotificationType': 'ACTUAL',
                    'ComparisonOperator': 'GREATER_THAN',
                    'Threshold': threshold_percent,
                    'ThresholdType': 'PERCENTAGE',
                    'NotificationState': 'ALARM'
                }
                
                subscriber = {
                    'SubscriptionType': 'EMAIL',
                    'Address': notification_email
                }
                
                budgets_client.create_notification(
                    AccountId=self.session.client('sts').get_caller_identity()['Account'],
                    BudgetName=budget_name,
                    Notification=notification,
                    Subscribers=[subscriber]
                )
            
            return response
        except Exception as e:
            logger.error(f"Error creating cost budget: {e}")
            return {}
    
    def analyze_costs_by_service(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        granularity: str = 'DAILY'
    ) -> pd.DataFrame:
        """
        Analyze costs by service and return a DataFrame.
        
        Args:
            start_date: Start date in YYYY-MM-DD format. If None, defaults to 30 days ago.
            end_date: End date in YYYY-MM-DD format. If None, defaults to today.
            granularity: Time granularity (DAILY, MONTHLY, HOURLY).
            
        Returns:
            DataFrame containing cost data by service.
        """
        response = self.get_project_costs(start_date, end_date, granularity)
        
        # Extract cost data
        cost_data = []
        for result in response.get('ResultsByTime', []):
            time_period = result['TimePeriod']
            start = time_period['Start']
            
            for group in result.get('Groups', []):
                service = group['Keys'][0]
                amount = float(group['Metrics']['UnblendedCost']['Amount'])
                unit = group['Metrics']['UnblendedCost']['Unit']
                
                cost_data.append({
                    'Date': start,
                    'Service': service,
                    'Amount': amount,
                    'Unit': unit
                })
        
        # Create DataFrame
        if cost_data:
            df = pd.DataFrame(cost_data)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        else:
            return pd.DataFrame(columns=['Date', 'Service', 'Amount', 'Unit'])
    
    def plot_costs_by_service(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        granularity: str = 'DAILY',
        top_n: int = 5,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot costs by service.
        
        Args:
            start_date: Start date in YYYY-MM-DD format. If None, defaults to 30 days ago.
            end_date: End date in YYYY-MM-DD format. If None, defaults to today.
            granularity: Time granularity (DAILY, MONTHLY, HOURLY).
            top_n: Number of top services to show.
            figsize: Figure size as (width, height).
            
        Returns:
            Matplotlib figure containing the plot.
        """
        df = self.analyze_costs_by_service(start_date, end_date, granularity)
        
        if df.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No cost data available', ha='center', va='center')
            ax.set_title('Costs by Service')
            return fig
        
        # Get top N services by total cost
        top_services = df.groupby('Service')['Amount'].sum().nlargest(top_n).index
        
        # Filter for top services
        df_top = df[df['Service'].isin(top_services)]
        
        # Create pivot table
        pivot_df = df_top.pivot_table(
            index='Date',
            columns='Service',
            values='Amount',
            aggfunc='sum'
        ).fillna(0)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        pivot_df.plot(kind='area', stacked=True, ax=ax, alpha=0.7)
        
        ax.set_title(f'Top {top_n} Services by Cost')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cost (USD)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_cost_breakdown(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot cost breakdown by service.
        
        Args:
            start_date: Start date in YYYY-MM-DD format. If None, defaults to 30 days ago.
            end_date: End date in YYYY-MM-DD format. If None, defaults to today.
            figsize: Figure size as (width, height).
            
        Returns:
            Matplotlib figure containing the plot.
        """
        df = self.analyze_costs_by_service(start_date, end_date, 'MONTHLY')
        
        if df.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No cost data available', ha='center', va='center')
            ax.set_title('Cost Breakdown by Service')
            return fig
        
        # Aggregate by service
        service_costs = df.groupby('Service')['Amount'].sum().sort_values(ascending=False)
        
        # Create pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Pie chart
        ax1.pie(
            service_costs,
            labels=service_costs.index,
            autopct='%1.1f%%',
            startangle=90,
            shadow=False
        )
        ax1.axis('equal')
        ax1.set_title('Cost Breakdown by Service')
        
        # Bar chart
        service_costs.plot(kind='bar', ax=ax2)
        ax2.set_title('Service Costs (USD)')
        ax2.set_ylabel('Cost (USD)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_cost_report(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_format: str = 'html',
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive cost report.
        
        Args:
            start_date: Start date in YYYY-MM-DD format. If None, defaults to 30 days ago.
            end_date: End date in YYYY-MM-DD format. If None, defaults to today.
            output_format: Output format ('html', 'markdown', 'json').
            output_file: Output file path. If None, the report is returned as a string.
            
        Returns:
            Report as a string in the specified format.
        """
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Get cost data
        project_costs = self.get_project_costs(start_date, end_date, 'MONTHLY')
        sagemaker_costs = self.get_sagemaker_costs_by_resource(start_date, end_date, 'MONTHLY')
        cost_forecast = self.get_cost_forecast(
            datetime.now().strftime('%Y-%m-%d'),
            (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        )
        
        # Calculate total cost
        total_cost = 0
        for result in project_costs.get('ResultsByTime', []):
            for group in result.get('Groups', []):
                total_cost += float(group['Metrics']['UnblendedCost']['Amount'])
        
        # Calculate SageMaker cost
        sagemaker_cost = 0
        for result in sagemaker_costs.get('ResultsByTime', []):
            for group in result.get('Groups', []):
                sagemaker_cost += float(group['Metrics']['UnblendedCost']['Amount'])
        
        # Calculate forecast
        forecast_total = 0
        for result in cost_forecast.get('ForecastResultsByTime', []):
            forecast_total += float(result['MeanValue'])
        
        # Create report data
        report_data = {
            'project_name': self.project_name,
            'start_date': start_date,
            'end_date': end_date,
            'total_cost': total_cost,
            'sagemaker_cost': sagemaker_cost,
            'forecast_total': forecast_total,
            'project_costs': project_costs,
            'sagemaker_costs': sagemaker_costs,
            'cost_forecast': cost_forecast
        }
        
        # Generate report in the specified format
        if output_format == 'json':
            report = json.dumps(report_data, indent=2, default=str)
        elif output_format == 'markdown':
            report = self._generate_markdown_report(report_data)
        else:  # html
            report = self._generate_html_report(report_data)
        
        # Write to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report
    
    def _generate_markdown_report(self, report_data: Dict) -> str:
        """
        Generate a markdown cost report.
        
        Args:
            report_data: Report data dictionary.
            
        Returns:
            Markdown report as a string.
        """
        report = f"# Cost Report for {report_data['project_name']}\n\n"
        report += f"**Period:** {report_data['start_date']} to {report_data['end_date']}\n\n"
        
        report += "## Summary\n\n"
        report += f"- **Total Cost:** ${report_data['total_cost']:.2f}\n"
        report += f"- **SageMaker Cost:** ${report_data['sagemaker_cost']:.2f} ({(report_data['sagemaker_cost'] / report_data['total_cost'] * 100) if report_data['total_cost'] > 0 else 0:.1f}% of total)\n"
        report += f"- **Forecast (Next 30 Days):** ${report_data['forecast_total']:.2f}\n\n"
        
        report += "## Cost Breakdown by Service\n\n"
        report += "| Service | Cost (USD) | Percentage |\n"
        report += "|---------|------------|------------|\n"
        
        service_costs = {}
        for result in report_data['project_costs'].get('ResultsByTime', []):
            for group in result.get('Groups', []):
                service = group['Keys'][0]
                amount = float(group['Metrics']['UnblendedCost']['Amount'])
                
                if service in service_costs:
                    service_costs[service] += amount
                else:
                    service_costs[service] = amount
        
        for service, amount in sorted(service_costs.items(), key=lambda x: x[1], reverse=True):
            percentage = (amount / report_data['total_cost'] * 100) if report_data['total_cost'] > 0 else 0
            report += f"| {service} | ${amount:.2f} | {percentage:.1f}% |\n"
        
        report += "\n## SageMaker Cost Breakdown by Resource\n\n"
        report += "| Resource | Cost (USD) | Percentage |\n"
        report += "|----------|------------|------------|\n"
        
        resource_costs = {}
        for result in report_data['sagemaker_costs'].get('ResultsByTime', []):
            for group in result.get('Groups', []):
                resource = group['Keys'][0]
                amount = float(group['Metrics']['UnblendedCost']['Amount'])
                
                if resource in resource_costs:
                    resource_costs[resource] += amount
                else:
                    resource_costs[resource] = amount
        
        for resource, amount in sorted(resource_costs.items(), key=lambda x: x[1], reverse=True):
            percentage = (amount / report_data['sagemaker_cost'] * 100) if report_data['sagemaker_cost'] > 0 else 0
            report += f"| {resource} | ${amount:.2f} | {percentage:.1f}% |\n"
        
        report += "\n## Cost Forecast (Next 30 Days)\n\n"
        report += "| Period | Forecast (USD) |\n"
        report += "|--------|---------------|\n"
        
        for result in report_data['cost_forecast'].get('ForecastResultsByTime', []):
            start = result['TimePeriod']['Start']
            end = result['TimePeriod']['End']
            amount = float(result['MeanValue'])
            
            report += f"| {start} to {end} | ${amount:.2f} |\n"
        
        report += "\n## Recommendations\n\n"
        
        # Add recommendations based on cost data
        if report_data['sagemaker_cost'] / report_data['total_cost'] > 0.5 and report_data['total_cost'] > 0:
            report += "- **SageMaker Costs:** SageMaker accounts for more than 50% of your total costs. Consider using spot instances for training jobs and auto-scaling for endpoints.\n"
        
        if report_data['forecast_total'] > report_data['total_cost'] * 1.2:
            report += "- **Cost Forecast:** Your forecasted costs are 20% higher than your current costs. Review your resource usage and consider implementing cost controls.\n"
        
        report += "- **General:** Regularly review your costs and implement cost allocation tags to track costs by feature or team.\n"
        
        report += "\n*Report generated on " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*\n"
        
        return report
    
    def _generate_html_report(self, report_data: Dict) -> str:
        """
        Generate an HTML cost report.
        
        Args:
            report_data: Report data dictionary.
            
        Returns:
            HTML report as a string.
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cost Report for {report_data['project_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #0066cc; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary {{ background-color: #e6f2ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .recommendations {{ background-color: #fff2e6; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .footer {{ font-size: 0.8em; color: #666; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <h1>Cost Report for {report_data['project_name']}</h1>
            <p><strong>Period:</strong> {report_data['start_date']} to {report_data['end_date']}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Cost:</strong> ${report_data['total_cost']:.2f}</p>
                <p><strong>SageMaker Cost:</strong> ${report_data['sagemaker_cost']:.2f} ({(report_data['sagemaker_cost'] / report_data['total_cost'] * 100) if report_data['total_cost'] > 0 else 0:.1f}% of total)</p>
                <p><strong>Forecast (Next 30 Days):</strong> ${report_data['forecast_total']:.2f}</p>
            </div>
            
            <h2>Cost Breakdown by Service</h2>
            <table>
                <tr>
                    <th>Service</th>
                    <th>Cost (USD)</th>
                    <th>Percentage</th>
                </tr>
        """
        
        service_costs = {}
        for result in report_data['project_costs'].get('ResultsByTime', []):
            for group in result.get('Groups', []):
                service = group['Keys'][0]
                amount = float(group['Metrics']['UnblendedCost']['Amount'])
                
                if service in service_costs:
                    service_costs[service] += amount
                else:
                    service_costs[service] = amount
        
        for service, amount in sorted(service_costs.items(), key=lambda x: x[1], reverse=True):
            percentage = (amount / report_data['total_cost'] * 100) if report_data['total_cost'] > 0 else 0
            html += f"""
                <tr>
                    <td>{service}</td>
                    <td>${amount:.2f}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>SageMaker Cost Breakdown by Resource</h2>
            <table>
                <tr>
                    <th>Resource</th>
                    <th>Cost (USD)</th>
                    <th>Percentage</th>
                </tr>
        """
        
        resource_costs = {}
        for result in report_data['sagemaker_costs'].get('ResultsByTime', []):
            for group in result.get('Groups', []):
                resource = group['Keys'][0]
                amount = float(group['Metrics']['UnblendedCost']['Amount'])
                
                if resource in resource_costs:
                    resource_costs[resource] += amount
                else:
                    resource_costs[resource] = amount
        
        for resource, amount in sorted(resource_costs.items(), key=lambda x: x[1], reverse=True):
            percentage = (amount / report_data['sagemaker_cost'] * 100) if report_data['sagemaker_cost'] > 0 else 0
            html += f"""
                <tr>
                    <td>{resource}</td>
                    <td>${amount:.2f}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Cost Forecast (Next 30 Days)</h2>
            <table>
                <tr>
                    <th>Period</th>
                    <th>Forecast (USD)</th>
                </tr>
        """
        
        for result in report_data['cost_forecast'].get('ForecastResultsByTime', []):
            start = result['TimePeriod']['Start']
            end = result['TimePeriod']['End']
            amount = float(result['MeanValue'])
            
            html += f"""
                <tr>
                    <td>{start} to {end}</td>
                    <td>${amount:.2f}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <div class="recommendations">
                <h2>Recommendations</h2>
                <ul>
        """
        
        # Add recommendations based on cost data
        if report_data['sagemaker_cost'] / report_data['total_cost'] > 0.5 and report_data['total_cost'] > 0:
            html += "<li><strong>SageMaker Costs:</strong> SageMaker accounts for more than 50% of your total costs. Consider using spot instances for training jobs and auto-scaling for endpoints.</li>"
        
        if report_data['forecast_total'] > report_data['total_cost'] * 1.2:
            html += "<li><strong>Cost Forecast:</strong> Your forecasted costs are 20% higher than your current costs. Review your resource usage and consider implementing cost controls.</li>"
        
        html += "<li><strong>General:</strong> Regularly review your costs and implement cost allocation tags to track costs by feature or team.</li>"
        
        html += f"""
                </ul>
            </div>
            
            <div class="footer">
                <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
        return html


def create_cost_alert(
    metric_name: str,
    threshold: float,
    comparison_operator: str = 'GREATER_THAN',
    evaluation_periods: int = 1,
    notification_topic: str = None,
    session=None,
    profile_name: str = 'ab'
) -> Dict:
    """
    Create a CloudWatch alarm for cost metrics.
    
    Args:
        metric_name: Cost metric name.
        threshold: Alarm threshold.
        comparison_operator: Comparison operator.
        evaluation_periods: Number of evaluation periods.
        notification_topic: SNS topic ARN for notifications.
        session: Boto3 session to use. If None, a new session will be created.
        profile_name: AWS profile name to use if session is None.
        
    Returns:
        Dict containing the alarm creation response.
    """
    if session is None:
        session = boto3.Session(profile_name=profile_name)
    
    cloudwatch = session.client('cloudwatch')
    
    try:
        alarm_name = f"Cost-Alert-{metric_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        alarm_config = {
            'AlarmName': alarm_name,
            'AlarmDescription': f'Cost alert for {metric_name}',
            'ActionsEnabled': True,
            'MetricName': metric_name,
            'Namespace': 'AWS/Billing',
            'Statistic': 'Maximum',
            'Dimensions': [
                {
                    'Name': 'Currency',
                    'Value': 'USD'
                }
            ],
            'Period': 86400,  # 1 day
            'EvaluationPeriods': evaluation_periods,
            'Threshold': threshold,
            'ComparisonOperator': comparison_operator,
            'TreatMissingData': 'missing'
        }
        
        if notification_topic:
            alarm_config['AlarmActions'] = [notification_topic]
        
        response = cloudwatch.put_metric_alarm(**alarm_config)
        
        logger.info(f"Created cost alert: {alarm_name}")
        return {'AlarmName': alarm_name}
    except Exception as e:
        logger.error(f"Error creating cost alert: {e}")
        return {}


def create_cost_dashboard(
    dashboard_name: str,
    session=None,
    profile_name: str = 'ab'
) -> Dict:
    """
    Create a CloudWatch dashboard for cost metrics.
    
    Args:
        dashboard_name: Dashboard name.
        session: Boto3 session to use. If None, a new session will be created.
        profile_name: AWS profile name to use if session is None.
        
    Returns:
        Dict containing the dashboard creation response.
    """
    if session is None:
        session = boto3.Session(profile_name=profile_name)
    
    cloudwatch = session.client('cloudwatch')
    
    try:
        # Create dashboard body
        dashboard_body = {
            'widgets': [
                {
                    'type': 'metric',
                    'x': 0,
                    'y': 0,
                    'width': 12,
                    'height': 6,
                    'properties': {
                        'metrics': [
                            ['AWS/Billing', 'EstimatedCharges', 'Currency', 'USD']
                        ],
                        'period': 86400,
                        'stat': 'Maximum',
                        'region': 'us-east-1',
                        'title': 'Estimated Charges'
                    }
                },
                {
                    'type': 'metric',
                    'x': 0,
                    'y': 6,
                    'width': 12,
                    'height': 6,
                    'properties': {
                        'metrics': [
                            ['AWS/Billing', 'EstimatedCharges', 'ServiceName', 'AmazonSageMaker', 'Currency', 'USD']
                        ],
                        'period': 86400,
                        'stat': 'Maximum',
                        'region': 'us-east-1',
                        'title': 'SageMaker Charges'
                    }
                },
                {
                    'type': 'metric',
                    'x': 12,
                    'y': 0,
                    'width': 12,
                    'height': 6,
                    'properties': {
                        'metrics': [
                            ['AWS/Billing', 'EstimatedCharges', 'ServiceName', 'AmazonS3', 'Currency', 'USD']
                        ],
                        'period': 86400,
                        'stat': 'Maximum',
                        'region': 'us-east-1',
                        'title': 'S3 Charges'
                    }
                },
                {
                    'type': 'metric',
                    'x': 12,
                    'y': 6,
                    'width': 12,
                    'height': 6,
                    'properties': {
                        'metrics': [
                            ['AWS/Billing', 'EstimatedCharges', 'ServiceName', 'AmazonCloudWatch', 'Currency', 'USD']
                        ],
                        'period': 86400,
                        'stat': 'Maximum',
                        'region': 'us-east-1',
                        'title': 'CloudWatch Charges'
                    }
                }
            ]
        }
        
        response = cloudwatch.put_dashboard(
            DashboardName=dashboard_name,
            DashboardBody=json.dumps(dashboard_body)
        )
        
        logger.info(f"Created cost dashboard: {dashboard_name}")
        return {'DashboardName': dashboard_name}
    except Exception as e:
        logger.error(f"Error creating cost dashboard: {e}")
        return {}