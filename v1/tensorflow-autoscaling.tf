# Required provider block for AWS
provider "aws" {
  region = "us-west-2"  # Change to your desired region
}

# Create a new VPC
resource "aws_vpc" "my_vpc" {
  cidr_block = "10.0.0.0/16"
}

# Create a new subnet inside the VPC
resource "aws_subnet" "my_subnet" {
  vpc_id                  = aws_vpc.my_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "us-west-2a"  # Change to your desired availability zone
}

# Create a new security group
resource "aws_security_group" "my_security_group" {
  vpc_id      = aws_vpc.my_vpc.id
  
  # Inbound rules
  ingress {
    from_port   = 22  # SSH
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Allow SSH access from anywhere
  }
  
  # Outbound rules
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]  # Allow all outbound traffic
  }
}

# Create a new S3 bucket for logs/artifacts
resource "aws_s3_bucket" "my_bucket" {
  bucket = "my-output-bucket"  # Change to your desired bucket name
}

# Create an IAM role for the ASG instances
resource "aws_iam_role" "my_role" {
  name = "my-iam-role"  # Change to your desired role name
  
  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Effect": "Allow",
      "Sid": ""
    }
  ]
}
EOF
}

# Attach the AWS managed policy to the IAM role
resource "aws_iam_role_policy_attachment" "my_policy_attachment" {
  role       = aws_iam_role.my_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2RoleforAWSAutoScaling"
}

# Create an Auto Scaling Group
resource "aws_autoscaling_group" "my_asg" {
  vpc_zone_identifier        = [aws_subnet.my_subnet.id]
  launch_configuration      = aws_launch_configuration.my_launch_config.name
  min_size                  = 1
  max_size                  = 1
  desired_capacity          = 1
  health_check_type         = "EC2"
  termination_policies      = ["OldestInstance"]
}

# Create a Launch Configuration for the Auto Scaling Group
resource "aws_launch_configuration" "my_launch_config" {
  name                        = "my-launch-config"  # Change to your desired name
  image_id                    = "ami-12345678"  # Change to your desired image ID
  instance_type               = "t2.micro"  # Change to your desired instance type
  security_groups             = [aws_security_group.my_security_group.id]
  iam_instance_profile        = aws_iam_role.my_role.name
  enable_monitoring           = true
}

# Output the ASG details and S3 bucket details
output "autoscaling_group_name" {
  value = aws_autoscaling_group.my_asg.name
}

output "autoscaling_group_arn" {
  value = aws_autoscaling_group.my_asg.arn
}

output "autoscaling_group_min_size" {
  value = aws_autoscaling_group.my_asg.min_size
}

output "autoscaling_group_max_size" {
  value = aws_autoscaling_group.my_asg.max_size
}

output "autoscaling_group_desired_capacity" {
  value = aws_autoscaling_group.my_asg.desired_capacity
}

output "s3_bucket_name" {
  value = aws_s3_bucket.my_bucket.bucket
}

