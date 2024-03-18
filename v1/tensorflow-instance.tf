# Provider configuration
provider "aws" {
  #region     = "us-east-1"
  #access_key = "<your_aws_access_key>"
  #secret_key = "<your_aws_secret_key>"
}

# Create VPC
resource "aws_vpc" "example_vpc" {
  cidr_block = "10.0.0.0/16"
}

# Create subnet within VPC
resource "aws_subnet" "example_subnet" {
  vpc_id     = aws_vpc.example_vpc.id
  cidr_block = "10.0.0.0/24"
}

# Create Security Group
resource "aws_security_group" "example_sg" {
  name        = "example_sg"
  description = "Example Security Group"
  vpc_id      = aws_vpc.example_vpc.id
  
  # Inbound rule
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # Outbound rule
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Create S3 bucket to store logs/artifacts
resource "aws_s3_bucket" "example_bucket" {
  bucket = "example-bucket"
  acl    = "private"
}

# Create EC2 instance
resource "aws_instance" "example_instance" {
  ami           = "ami-xxxxxxxx"  # Replace with your desired AMI ID
  instance_type = "t2.micro"
  subnet_id     = aws_subnet.example_subnet.id
  vpc_security_group_ids = [aws_security_group.example_sg.id]
  
  user_data = <<EOF
#cloud-config
runcmd:
  - wget -O https://github.com/nand0p/ai/tensorflow.cloudinit.sh /root/tensorflow.cloudinit.sh
  - chmod -c 0755 /root/tensorflow.cloudinit.sh
  - bash /root/tensorflow.cloudinit.sh
EOF


  provisioner "local-exec" {
    command = "terraform output -json > instance-data.json"
  }

  provisioner "remote-exec" {
    inline = [
      "cat > tensorflow.cloudinit.yaml <<EOF",
      "${file("${path.module}/tensorflow.cloudinit.yaml")}",
      "EOF",
      "sudo cp tensorflow.cloudinit.yaml /var/lib/cloud/instances/${self.id}/user-data.txt",
    ]
  }
  
  # Output logs to S3 bucket
  lifecycle {
    create_before_destroy = true
  }

  ebs_block_device {
    device_name = "/dev/sda1"
    volume_type = "gp2"
    volume_size = 8
  }

  root_block_device {
    volume_type           = "gp2"
    volume_size           = 8
    delete_on_termination = true
  }
  
  # Store logs in S3
  userData = base64encode(render("logs.tf"))
  
  metadata_opts {
    http_endpoint = "enabled"  # Required for cloud-init to work
  }

  tags = {
    Name = "tensorflow-training"
  }
}

